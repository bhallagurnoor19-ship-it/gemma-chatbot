Set fso = CreateObject("Scripting.FileSystemObject")
Set WshShell = CreateObject("WScript.Shell")

strPath = fso.GetParentFolderName(WScript.ScriptFullName)
strPythonw = "C:\Users\Gurnoor\AppData\Local\Programs\Python\Python313\pythonw.exe"

If Not fso.FileExists(strPythonw) Then
    strPythonw = "pythonw.exe"
End If

' Run launcher silently with hidden window (0)
WshShell.Run """" & strPythonw & """ """ & strPath & "\run_app.py""", 0, False

Set fso = Nothing
Set WshShell = Nothing
