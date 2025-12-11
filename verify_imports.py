import importlib
packages = [
    'streamlit','requests','pdf2image','langchain_community','langchain','langchain_huggingface',
    'langchain_community.vectorstores','langchain_ollama','langchain.chains','streamlit_mic_recorder','pyttsx3','json','datetime','uuid'
]
missing = []
for p in packages:
    try:
        importlib.import_module(p)
    except Exception as e:
        missing.append((p, str(e)))
print('Missing or import errors:')
if not missing:
    print('None')
else:
    for m in missing:
        print(m[0],'->',m[1])
