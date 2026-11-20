$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$sdkHeader = Join-Path $projectRoot 'build/ort-sdk/include/onnxruntime_cxx_api.h'
if (-not (Test-Path -LiteralPath $sdkHeader)) {
    throw 'Prepare the SDK with python native/fetch_sdk.py first.'
}
$cmakeCommand = Get-Command cmake -ErrorAction SilentlyContinue
if ($cmakeCommand) {
    $cmakeExe = $cmakeCommand.Source
} else {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    if (-not (Test-Path -LiteralPath $vswhere)) {
        throw 'The Visual Studio Installer or CMake is required.'
    }
    $cmakeExe = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' | Select-Object -First 1
    if (-not $cmakeExe) { throw 'Install the CMake component for Visual Studio 2022.' }
}
$buildDir = Join-Path $projectRoot 'build/native'
& $cmakeExe -S $PSScriptRoot -B $buildDir -G 'Visual Studio 17 2022' -A x64
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $cmakeExe --build $buildDir --config Release
exit $LASTEXITCODE
