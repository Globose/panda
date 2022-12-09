import hashlib as hl

with open('C:/Users/Tobias/Downloads/windows_10_cmake_Release_graphviz-install-6.0.2-win64.exe', 'rb') as f:
    file = f.read()
    hash = hl.sha256(file).hexdigest()
    print(hash)


