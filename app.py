from flask import Flask, render_template, request, redirect, url_for, send_file, session, send_from_directory
import os
import pandas as pd
import pdfkit
from jinja2 import Template
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from validation_script import make_report
import secrets
secret_key = secrets.token_hex(16)

app = Flask(__name__, static_folder='outputs')

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.secret_key = secret_key

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Helper functions here (e.g., processing data, generating reports)
def process_files(file1_path, file2_path):
    # Placeholder processing function (replace with your actual code logic)
    # Create mock data for HTML and PDF reports as examples
    html_report_path = os.path.join(app.config['OUTPUT_FOLDER'], "validation_report.html")
    pdf_report_path = os.path.join(app.config['OUTPUT_FOLDER'], "validation_report.pdf")

    # Generate a simple HTML report as a demo
    html_content = "<h1>Report</h1><p>Report generated based on uploaded files.</p>"
    with open(html_report_path, "w") as f:
        f.write(html_content)

    # Convert HTML report to PDF
    pdfkit.from_file(html_report_path, pdf_report_path)

    return html_report_path, pdf_report_path

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        old_file = request.files["old-file"]
        new_file = request.files["new-file"]
        if old_file and new_file:
            session['uploaded_filename'] = new_file.filename
            # Save the uploaded files
            old_file_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                         secure_filename(old_file.filename))
            new_file_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                         secure_filename(new_file.filename))
            old_file.save(old_file_path)
            new_file.save(new_file_path)

            # Process the files
            # Simulate processing time
            # import time
            # time.sleep(5)
            # html_report, pdf_report = process_files(new_file_path, old_file_path)
            output_folder = app.config['OUTPUT_FOLDER']
            html_report, pdf_report = make_report(new_file_path=new_file_path,
                                                  old_file_path=old_file_path,
                                                  output_folder=output_folder)
            
            # Delete the uploaded files from the server
            try:
                os.remove(new_file_path)
                os.remove(old_file_path)
            except OSError as e:
                print(f"Error deleting files: {e}")

            # Send back the HTML and PDF files to the index page
            return render_template("index.html", 
                                   html_report=html_report,
                                   pdf_report=pdf_report)

    return render_template("index.html")

@app.route("/view_html")
def view_html():
    html_report_path = os.path.join(app.config['OUTPUT_FOLDER'], "validation_report.html")
    if os.path.exists(html_report_path):
        with open(html_report_path, 'r') as file:
            html_content = file.read()
        return html_content
    else:
        return "HTML report not found", 404


@app.route("/download_pdf")
def download_pdf():
    pdf_report_path = os.path.join(app.config['OUTPUT_FOLDER'], "validation_report.pdf")
    
    # Get the name of the uploaded file (you'll need to store this when uploading)
    uploaded_filename = session.get('uploaded_filename', 'unknown')
    
    # Get the base name of the original file (without extension)
    original_name = os.path.splitext(uploaded_filename)[0]
    
    # Create a new filename with the original name as a suffix
    new_filename = f"validation_report_{original_name}.pdf"
    
    return send_file(pdf_report_path, as_attachment=True, download_name=new_filename)



@app.route('/outputs/<path:filename>')
def serve_output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
