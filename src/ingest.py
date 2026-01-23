import docling
import os

from docling.document_converter import DocumentConverter


#source = r"data/01-11-2020-203418Zero to One.pdf"
path = r"/Users/lekeadako/Documents/portfolio projects/Book_chat/books"

def convet_to_md(path):
    for file_name in os.listdir(path):
        full_path = os.path.join(path,file_name)

        if os.path.isfile(full_path):
            exten = os.path.splitext(file_name)[1].lower()
            filename = os.path.splitext(file_name)[0].lower()

            converter = DocumentConverter()
            if exten == '.pdf':
        
                doc = converter.convert(full_path).document
                md_text = doc.export_to_markdown()
                output_path = os.path.join(r'/Users/lekeadako/Documents/portfolio projects/Book_chat/data',filename + '.md')

                try:
                    with open(output_path, 'w', encoding='utf-8') as file:
                        file.write(md_text)
                    print(f"Successfully saved content to {filename}")
                except IOError as e:
                    print(f"Error saving file: {e}")
                


convet_to_md(path)