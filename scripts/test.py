from data_utils import ASSETS_PATH
from docx import Document

# word文档的位置
file = ASSETS_PATH + 'test.docx'

document = Document(file)

def cut_doc_out(document,head,tail):
    paras = []
    start = False
    for paragraph in document.paragraphs:
        if head in paragraph.text:
            print('有开头')
            new_paragraphs = []
            new_paragraphs.append(paragraph)
            start = True
        if tail in paragraph.text:
            print('有结尾')
            paras.append(new_paragraphs)
            start = False
        if start == True:
            print('有中间')
            new_paragraphs.append(paragraph)
    return paras

def get_para_data(output_doc_name, paragraph):
    """
    Write the run to the new file and then set its font, bold, alignment, color etc. data.
    """

    output_para = output_doc_name.add_paragraph()
    for run in paragraph.runs:
        output_run = output_para.add_run(run.text)
        # Run's bold data
        output_run.bold = run.bold
        # Run's italic data
        output_run.italic = run.italic
        # Run's underline data
        output_run.underline = run.underline
        # Run's color data
        output_run.font.color.rgb = run.font.color.rgb
        # Run's font data
        output_run.style.name = run.style.name
        # Run's font size
        output_run.font.size = run.font.size
    # Paragraph's alignment data
    output_para.style = paragraph.style
    output_para.alignment = paragraph.alignment
    output_para.paragraph_format.alignment = paragraph.paragraph_format.alignment
    output_para.paragraph_format.widow_control = paragraph.paragraph_format.widow_control


start = '投标函及投标函附录'
end = '法定代表人身份证明'

paras = cut_doc_out(document,start,end)

new_doc = Document()

for para in paras[1]:
    get_para_data(new_doc,para)

new_doc.save('test.docx')

