; OpenOne Windows installer (Inno Setup 6). Built by packaging/windows/build.ps1:
;   ISCC.exe /DAppVersion=0.3.0 /DDistDir=..\..\dist\OpenOne /DOutDir=..\..\release /DSuffix= openone.iss
; Per-user install (no admin needed); the model package is NOT included, it is imported at first start.
#ifndef AppVersion
  #define AppVersion "0.3.0"
#endif
#ifndef DistDir
  #define DistDir "..\..\dist\OpenOne"
#endif
#ifndef OutDir
  #define OutDir "..\..\release"
#endif
#ifndef Suffix
  #define Suffix ""
#endif

[Setup]
AppId={{6F1C2B8E-4D0A-4A57-9C6B-0E5D7A1F0A11}
AppName=OpenOne
AppVersion={#AppVersion}
AppPublisher=OpenOne (private build)
AppComments=Local tester for the OpenOne decision model. Model weights are imported separately.
DefaultDirName={localappdata}\Programs\OpenOne
DefaultGroupName=OpenOne
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir={#OutDir}
OutputBaseFilename=OpenOne-Setup-{#AppVersion}-win-x64{#Suffix}
SetupIconFile=..\assets\icon.ico
UninstallDisplayIcon={app}\OpenOne.exe
Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
MinVersion=10.0.17763

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "{#DistDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\OpenOne"; Filename: "{app}\OpenOne.exe"
Name: "{group}\Uninstall OpenOne"; Filename: "{uninstallexe}"
Name: "{autodesktop}\OpenOne"; Filename: "{app}\OpenOne.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\OpenOne.exe"; Description: "Start OpenOne"; Flags: nowait postinstall skipifsilent

[Code]
function WebView2Installed(): Boolean;
var v: String;
begin
  Result := RegQueryStringValue(HKLM, 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', v)
    or RegQueryStringValue(HKCU, 'Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', v);
  Result := Result and (v <> '') and (v <> '0.0.0.0');
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep = ssPostInstall) and (not WebView2Installed()) then
    MsgBox('OpenOne needs the Microsoft Edge WebView2 Runtime (included in Windows 11). It was not found; install it from https://go.microsoft.com/fwlink/p/?LinkId=2124703', mbInformation, MB_OK);
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var data: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    data := ExpandConstant('{localappdata}\OpenOne');
    if DirExists(data) then
      if MsgBox('Also delete the imported models, runtimes, logs and feedback in' + #13#10 + data + ' ?', mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
        DelTree(data, True, True, True);
  end;
end;
