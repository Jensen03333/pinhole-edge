#define NOMINMAX
#include <windows.h>
#include <wincodec.h>
#include <wrl/client.h>
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;
constexpr int width = 640, height = 480, side = 512, scaledHeight = 384, padTop = 64;

void checked(HRESULT result, const char* message) {
    if (FAILED(result)) throw std::runtime_error(message);
}

std::vector<BYTE> loadRgb(IWICImagingFactory* factory, const wchar_t* path) {
    ComPtr<IWICBitmapDecoder> decoder;
    checked(factory->CreateDecoderFromFilename(path, nullptr, GENERIC_READ, WICDecodeMetadataCacheOnLoad,
            decoder.GetAddressOf()), "Cannot open image.");
    ComPtr<IWICBitmapFrameDecode> frame;
    checked(decoder->GetFrame(0, frame.GetAddressOf()), "Cannot decode frame 0.");
    UINT w = 0, h = 0;
    checked(frame->GetSize(&w, &h), "Cannot read image dimensions.");
    if (w != width || h != height) throw std::runtime_error("Expected a 640 x 480 source frame.");
    ComPtr<IWICFormatConverter> rgb;
    checked(factory->CreateFormatConverter(rgb.GetAddressOf()), "Cannot create RGB converter.");
    checked(rgb->Initialize(frame.Get(), GUID_WICPixelFormat24bppRGB, WICBitmapDitherTypeNone,
            nullptr, 0.0, WICBitmapPaletteTypeCustom), "Cannot convert frame to RGB.");
    std::vector<BYTE> pixels(width * height * 3);
    checked(rgb->CopyPixels(nullptr, width * 3, static_cast<UINT>(pixels.size()), pixels.data()),
            "Cannot read RGB pixels.");
    return pixels;
}

std::vector<float> preprocess(const std::vector<BYTE>& rgb) {
    // At this fixture's 4/5 scale, bilinear source coordinates have eighth-pixel precision.
    // The two integer truncations match OpenCV INTER_LINEAR on 8-bit frames.
    std::vector<float> tensor(3 * side * side, 114.0f / 255.0f);
    for (int y = 0; y < scaledHeight; ++y) {
        const int y8 = 10*y + 1, sy = y8/8, fy = y8%8;
        for (int x = 0; x < side; ++x) {
            const int x8 = 10*x + 1, sx = x8/8, fx = x8%8;
            const int at = (sy*width + sx)*3;
            for (int channel = 0; channel < 3; ++channel) {
                const int top = (8-fx)*rgb[at+channel] + fx*rgb[at+3+channel];
                const int bottom = (8-fx)*rgb[at+width*3+channel] + fx*rgb[at+width*3+3+channel];
                const int pixel = (((8-fy)*top/16) + (fy*bottom/16) + 2)/4;
                tensor[channel*side*side + (y+padTop)*side + x] = pixel / 255.0f;
            }
        }
    }
    return tensor;
}

struct Box { double left, top, right, bottom; float score; };

double overlap(const Box& a, const Box& b) {
    const double w = std::max(0.0, std::min(a.right,b.right)-std::max(a.left,b.left));
    const double h = std::max(0.0, std::min(a.bottom,b.bottom)-std::max(a.top,b.top));
    return w*h/((a.right-a.left)*(a.bottom-a.top)+(b.right-b.left)*(b.bottom-b.top)-w*h);
}

std::vector<Box> decode(const float* raw, size_t n, float cutoff) {
    std::vector<Box> candidates;
    auto xPixel = [](float x){ return std::clamp(static_cast<double>(x)/0.8, 0.0, double(width)); };
    auto yPixel = [](float y){ return std::clamp((static_cast<double>(y)-padTop)/0.8, 0.0, double(height)); };
    for (size_t i = 0; i < n; ++i) {
        if (raw[4*n+i] < cutoff) continue;
        const float cx = raw[i], cy = raw[n+i], halfW = raw[2*n+i]/2, halfH = raw[3*n+i]/2;
        Box b{xPixel(cx-halfW),yPixel(cy-halfH),xPixel(cx+halfW),yPixel(cy+halfH),raw[4*n+i]};
        if (b.right > b.left && b.bottom > b.top) candidates.push_back(b);
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](const Box& a, const Box& b){ return a.score > b.score; });
    std::vector<Box> kept;
    for (const Box& b : candidates) {
        if (std::none_of(kept.begin(), kept.end(), [&](const Box& k){ return overlap(b,k) > 0.5; }))
            kept.push_back(b);
        if (kept.size() == 200) break;
    }
    return kept;
}

void inspect(int argc, wchar_t** argv) {
    std::wstring thresholdArg(argv[2]);
    size_t end = 0;
    const float threshold = std::stof(thresholdArg, &end);
    if (end != thresholdArg.size() || !std::isfinite(threshold) || threshold <= 0 || threshold > 1)
        throw std::runtime_error("Confidence must be in (0, 1].");
    ComPtr<IWICImagingFactory> factory;
    checked(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                             IID_PPV_ARGS(factory.GetAddressOf())), "Cannot initialize Windows Imaging Component.");
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "pinhole-inspect");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(2);
    opts.SetInterOpNumThreads(1);
    Ort::Session session(env, argv[1], opts);
    const std::vector<int64_t> inputShape{1,3,side,side};
    if (session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape() != inputShape)
        throw std::runtime_error("This executable requires the static 512 x 512 single-frame model.");
    Ort::AllocatorWithDefaultOptions alloc;
    auto inputName = session.GetInputNameAllocated(0, alloc), outputName = session.GetOutputNameAllocated(0, alloc);
    const char* inputs[] = {inputName.get()};
    const char* outputs[] = {outputName.get()};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::cout << std::setprecision(10);
    for (int i = 3; i < argc; ++i) {
        auto values = preprocess(loadRgb(factory.Get(), argv[i]));
        auto tensor = Ort::Value::CreateTensor<float>(memory, values.data(), values.size(),
                                                        inputShape.data(), inputShape.size());
        auto results = session.Run(Ort::RunOptions{nullptr}, inputs, &tensor, 1, outputs, 1);
        auto info = results[0].GetTensorTypeAndShapeInfo();
        auto shape = info.GetShape();
        if (shape.size()!=3 || shape[0]!=1 || shape[1]!=5 || shape[2]<=0 ||
            info.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            throw std::runtime_error("Expected float output [1,5,anchors].");
        auto boxes = decode(results[0].GetTensorData<float>(), static_cast<size_t>(shape[2]), threshold);
        std::cout << "{\"index\":" << i-3 << ",\"boxes\":[";
        for (size_t j = 0; j < boxes.size(); ++j) {
            const auto& b = boxes[j];
            if (j) std::cout << ',';
            std::cout << '[' << b.left << ',' << b.top << ',' << b.right << ',' << b.bottom << ',' << b.score << ']';
        }
        std::cout << "]}\n";
    }
}

int wmain(int argc, wchar_t** argv) {
    if (argc < 4) {
        std::cerr << "Usage: pinhole-inspect.exe model.onnx threshold frame1 [frame2 ...]\n";
        return 2;
    }
    if (FAILED(CoInitializeEx(nullptr, COINIT_MULTITHREADED))) return 1;
    int status = 0;
    try { inspect(argc, argv); }
    catch (const std::exception& err) { std::cerr << err.what() << '\n'; status = 1; }
    CoUninitialize();
    return status;
}
