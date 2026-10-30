"""生成确定的中英文OCR验收素材，不使用模型生成预期答案。"""
from pathlib import Path

from PIL import Image,ImageDraw,ImageFont
from pypdf import PdfReader,PdfWriter

root=Path(__file__).resolve().parents[1]
output=root/"results/ocr-fixtures"
output.mkdir(parents=True,exist_ok=True)
font_path=Path("C:/Windows/Fonts/msyh.ttc")
if not font_path.is_file():
    raise RuntimeError("Set a CJK font path for this optional Windows acceptance fixture")
font=ImageFont.truetype(str(font_path),46)
heading=ImageFont.truetype(str(font_path),58)

def page(lines):
    image=Image.new("RGB",(1600,1050),"white")
    drawing=ImageDraw.Draw(image)
    for index,line in enumerate(lines):
        drawing.text((85,70+index*115),line,font=heading if index==0 else font,fill="black")
    return image

first=page(["TraceRAG OCR 实验记录","实验编号：TR-2026-108","样品：钙钛矿太阳能电池",
            "测试温度：25 C","光电转换效率：23.5%","Conclusion: Evidence must be traceable."])
second=page(["Laboratory Scan - Page Two","Batch identifier: ZEBRA-482",
             "The catalyst temperature is 42 C.","The measured pressure is 3.6 kPa.",
             "OCR transcription needs source verification."])
first.save(output/"chinese-english.png")
second.save(output/"english.png")
first.save(output/"scanned-lab.pdf",save_all=True,append_images=[second],resolution=144)
first.close(); second.close()
reader=PdfReader(output/"scanned-lab.pdf")
assert len(reader.pages)==2 and all(not p.extract_text().strip() for p in reader.pages)
writer=PdfWriter()
writer.add_page(reader.pages[0])
writer.add_blank_page(width=800,height=525)
writer.add_page(reader.pages[1])
with (output/"scan-with-blank.pdf").open("wb") as stream:
    writer.write(stream)
print(output)
