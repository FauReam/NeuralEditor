with open('src/web/romance.html','r',encoding='utf-8') as f:
    content = f.read()
old = '''  if (res.ok) {
    if (res.affection_delta) {'''
new = '''  if (res.ok) {
    if (res.response) addMsg('assistant', res.response);
    if (res.affection_delta) {'''
if old in content:
    content = content.replace(old, new)
    with open('src/web/romance.html','w',encoding='utf-8') as f:
        f.write(content)
    print('OK')
else:
    print('NOT FOUND')
