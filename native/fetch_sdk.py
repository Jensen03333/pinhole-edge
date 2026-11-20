"""Get the CPU development files for MSVC from Microsoft's versioned NuGet package."""
from pathlib import Path
import shutil
from tempfile import TemporaryFile
from urllib.request import urlopen
from zipfile import ZipFile

VERSION = "1.20.1"
URL = f"https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/{VERSION}/microsoft.ml.onnxruntime.{VERSION}.nupkg"
SDK = Path(__file__).resolve().parents[1] / "build" / "ort-sdk"


def fetch():
    with TemporaryFile() as package:
        with urlopen(URL, timeout=90) as source:
            shutil.copyfileobj(source, package, length=1024 * 1024)
        with ZipFile(package) as archive:
            for info in archive.infolist():
                path = Path(info.filename)
                lower = info.filename.lower()
                if lower.startswith("build/native/include/") and path.suffix == ".h":
                    dest = SDK / "include" / path.name
                elif lower.startswith("runtimes/win-x64/native/") and path.suffix in (".lib", ".dll"):
                    dest = SDK / "bin" / path.name
                elif path.name.lower() in {"license", "license.txt", "thirdpartynotices.txt"}:
                    dest = SDK / "notices" / path.name
                else:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, dest.open("wb") as output:
                    shutil.copyfileobj(source, output)
    for item in ("include/onnxruntime_cxx_api.h", "bin/onnxruntime.dll", "bin/onnxruntime.lib"):
        if not (SDK / item).is_file():
            raise RuntimeError(f"Missing {item} in the SDK")
    print("ONNX Runtime CPU SDK", VERSION, "ready")


if __name__ == "__main__":
    fetch()
