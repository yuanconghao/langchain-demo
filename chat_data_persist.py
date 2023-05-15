from utils import persist

contents = ['51talk', 'pdf', 'md']
for content in contents:
    if content == '51talk':
        persist.run_scan_51talk()
    elif content == 'pdf':
        persist.run_scan_pdfs()
    elif content == 'md':
        persist.run_scan_markdowns()
    else:
        print('nothing for now')