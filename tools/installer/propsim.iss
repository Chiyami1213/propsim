; PropSim Windows installer (Inno Setup 6, modern wizard). Built by tools/build_installer.py, which passes:
;   AppVersion, DistDir (PyInstaller one-folder build), OutDir, ArtDir (wizard bitmaps), IconFile, ReqFile, RepoURL, HelpURL

#define AppName "PropSim"
#define AppExe "PropSim.exe"
#ifndef Compress
  #define Compress "lzma2/ultra64"
#endif

[Setup]
AppId={{8E3C2A4B-5F1D-4C7A-9B2E-6D0F3A1C9E57}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=PropSim
AppPublisherURL={#RepoURL}
AppSupportURL={#HelpURL}
AppUpdatesURL={#RepoURL}/releases
AppComments=Interactive 3D rocket propulsion simulator
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
DisableWelcomePage=no
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
OutputDir={#OutDir}
OutputBaseFilename=PropSim-{#AppVersion}-Setup
SetupIconFile={#IconFile}
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName} {#AppVersion}
WizardStyle=modern
WizardImageFile={#ArtDir}\side_100.bmp,{#ArtDir}\side_125.bmp,{#ArtDir}\side_150.bmp,{#ArtDir}\side_175.bmp,{#ArtDir}\side_200.bmp,{#ArtDir}\side_225.bmp,{#ArtDir}\side_250.bmp
WizardSmallImageFile={#ArtDir}\small_100.bmp,{#ArtDir}\small_125.bmp,{#ArtDir}\small_150.bmp,{#ArtDir}\small_175.bmp,{#ArtDir}\small_200.bmp,{#ArtDir}\small_225.bmp,{#ArtDir}\small_250.bmp
InfoBeforeFile={#ReqFile}
Compression={#Compress}
SolidCompression=yes
LZMAUseSeparateProcess=yes
CloseApplications=yes
RestartApplications=no
ShowLanguageDialog=no
SetupLogging=yes
VersionInfoVersion={#AppVersion}
VersionInfoProductName={#AppName}
VersionInfoDescription={#AppName} Setup

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Messages]
WelcomeLabel2=This will install [name/ver] on your computer.%n%nPropSim is an interactive 3D simulator for liquid, hybrid, solid and electric rocket propulsion, with live parameter studies, time-domain runs, and code export to Python and MATLAB.%n%nIt is recommended that you close all other applications before continuing.
InfoBeforeLabel=Please check that this computer meets the requirements below before continuing.
FinishedLabel=Setup has finished installing [name] on your computer. The application may be launched by selecting the installed shortcuts.%n%nThe online help page (user guide, model notes and system requirements) is on the PropSim GitHub page.

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#DistDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autoprograms}\{#AppName}\{#AppName} Online Help"; Filename: "{#HelpURL}"
Name: "{autoprograms}\{#AppName}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent
Filename: "{#HelpURL}"; Description: "Open the {#AppName} help page on GitHub"; Flags: shellexec nowait postinstall skipifsilent

[Code]
type
  TMemoryStatusEx = record
    dwLength: Cardinal;
    dwMemoryLoad: Cardinal;
    ullTotalPhys: Int64;
    ullAvailPhys: Int64;
    ullTotalPageFile: Int64;
    ullAvailPageFile: Int64;
    ullTotalVirtual: Int64;
    ullAvailVirtual: Int64;
    ullAvailExtendedVirtual: Int64;
  end;

function GlobalMemoryStatusEx(var lpBuffer: TMemoryStatusEx): BOOL;
  external 'GlobalMemoryStatusEx@kernel32.dll stdcall';

function InitializeSetup(): Boolean;
var
  M: TMemoryStatusEx;
  MB: Int64;
begin
  Result := True;
  M.dwLength := SizeOf(M);
  if GlobalMemoryStatusEx(M) then
  begin
    MB := M.ullTotalPhys div 1048576;
    if MB < 3584 then
      Result := SuppressibleMsgBox('This computer has ' + IntToStr(MB) + ' MB of memory. PropSim needs at least 4 GB ' +
        '(8 GB recommended); large 3D simulations may fail.' + #13#10#13#10 + 'Continue anyway?',
        mbConfirmation, MB_YESNO, IDYES) = IDYES;
  end;
end;
