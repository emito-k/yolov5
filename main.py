import torch
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import os
import shutil
import textwrap
from datetime import datetime
import hashlib

models = [
    torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True),
    torch.hub.load('ultralytics/yolov5', 'custom', './runs/train/exp4/weights/best.pt', force_reload=True)
]

def add_text_to_pdf(c, text, x, y, max_width, line_height=14):
    wrapped_text = textwrap.fill(text, width=max_width)
    text_object = c.beginText(x, y)
    lines = wrapped_text.splitlines()
    for line in lines:
        text_object.textLine(line)
    c.drawText(text_object)
    # Calculate the final y position after the text block
    final_y = y - (len(lines) * line_height)
    return c, final_y

def add_bold_text_to_pdf(c, text, x, y):
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, text)
    return c

def add_image_to_pdf(c, img, x, y, width, height):
    c.drawImage(img, x, y, width, height)
    return c

def get_image_results(model, img):
    results = model(img)
    return results

def create_new_page(c):
    c.showPage()
    return c

def compute_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def process_file(file_url, output_pdf, counter, output_folder):
    input_md5 = compute_md5(file_url)
    output_location = ""
    for model in models:
        results = get_image_results(model, file_url)
        output_location = os.path.join(f"{output_folder}/output_{counter}.jpg")
        results.save()

        files = os.listdir("runs/detect")

        # get the latest folder where expn is the latest folder and n is the latest number
        expn = max([int(folder.split("exp")[1] or '0') for folder in files if "exp" in folder])

        file = os.listdir(f"runs/detect/exp{expn}/")[0]
        shutil.move(f"runs/detect/exp{expn}/{file}", output_location)

        output_md5 = compute_md5(output_location)
        output_pdf = create_new_page(output_pdf)

        # Adding file information and MD5 hashes
        output_pdf = add_bold_text_to_pdf(output_pdf, "Input File:", 10, 750)
        output_pdf, final_y = add_text_to_pdf(output_pdf, file_url, 10, 735, 80)

        output_pdf = add_bold_text_to_pdf(output_pdf, "Input File MD5:", 10, final_y - 15)
        output_pdf, final_y = add_text_to_pdf(output_pdf, input_md5, 10, final_y - 30, 80)

        output_pdf = add_bold_text_to_pdf(output_pdf, "Generated File:", 10, final_y - 15)
        output_pdf, final_y = add_text_to_pdf(output_pdf, output_location, 10, final_y - 30, 80)

        output_pdf = add_bold_text_to_pdf(output_pdf, "Generated File MD5:", 10, final_y - 15)
        output_pdf, final_y = add_text_to_pdf(output_pdf, output_md5, 10, final_y - 30, 80)

        # Adding scan results
        output_pdf = add_bold_text_to_pdf(output_pdf, "Scan results:", 10, final_y - 15)
        text_results = str(results.pandas().xyxy[0])
        output_pdf, final_y = add_text_to_pdf(output_pdf, text_results, 10, final_y - 30, 80)

        # Position the image below the text
        output_pdf = add_image_to_pdf(output_pdf, output_location, 10, final_y - 400, 400, 400)
        counter += 1

def add_cover_page(output_pdf):
    title = input("Enter the title of the document: (Object Detection Report)") or "Object Detection Report"
    author = input("Enter the author of the document: (Anonymous)") or "Anonymous"
    comment = input("Enter any comments: (This document contains the results of object detection on images)") or "This document contains the results of object detection on images"

    # Set font sizes
    output_pdf.setFont("Helvetica", 24)  # Title
    output_pdf.setFont("Helvetica", 16)  # Author
    output_pdf.setFont("Helvetica", 12)  # Comments and Scan Date

    # Title
    output_pdf.drawString(72, 700, title)

    # Author
    output_pdf.drawString(72, 670, f"Author: {author}")

    # Comments
    wrapped_comment = textwrap.fill(comment, width=60)
    lines = wrapped_comment.split('\n')
    y_coordinate = 650 - (len(lines) * 14)  # Adjust for multiple lines
    for line in lines:
        output_pdf.drawString(72, y_coordinate, f"Comments: {line}")
        y_coordinate -= 14  # Adjust for line spacing

    # Scan Date
    scan_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_pdf.drawString(72, y_coordinate - 14, f"Scan Date: {scan_date}")

    create_new_page(output_pdf)


def main():
    # output folder
    output_folder = input("Enter the output folder: (output)") or "output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    output_folder = os.path.abspath(output_folder)

    image_counter = 0
    folder = input("Input folder: ")

    pdf_output_name = input("Enter the PDF file name: (output.pdf)") or "output.pdf"
    pdf_output_path = os.path.join(output_folder, pdf_output_name)

    output_pdf = canvas.Canvas(pdf_output_path, pagesize=letter)

    # Add cover page
    add_cover_page(output_pdf)
    
    for file in os.listdir(folder):
        file_url = os.path.join(folder, file)
        process_file(file_url, output_pdf, image_counter, output_folder)
        image_counter += len(models) + 1

    output_pdf.save()

# if __name__ == "__main__":
main()