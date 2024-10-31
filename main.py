from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_file, session, send_from_directory, stream_with_context, Response
import os
from werkzeug.utils import secure_filename
# from validation_script_media_spend import make_validation_report
from validation_script_v2 import make_validation_report
import secrets
secret_key = secrets.token_hex(16)

app = Flask(__name__, static_folder='outputs')

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.secret_key = secret_key

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        old_file = request.files["old-file"]
        new_file = request.files["new-file"]
        country_name = request.form.get("country_name") or None
        start_date = request.form.get("start-date") or None
        end_date = request.form.get("end-date") or None
        if old_file and new_file:
            session['uploaded_filename'] = new_file.filename
            # Save the uploaded files
            old_file_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                         secure_filename(old_file.filename))
            new_file_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                         secure_filename(new_file.filename))
            old_file.save(old_file_path)
            new_file.save(new_file_path)

            output_folder = app.config['OUTPUT_FOLDER']

            html_report, pdf_report = make_validation_report(new_file_path=new_file_path,
                                                  old_file_path=old_file_path,
                                                  output_folder=output_folder,
                                                  country_name = country_name,
                                                  start_date = start_date,
                                                  end_date = end_date
                                                  )
            
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
    try:
        pdf_report_path = os.path.join(app.config['OUTPUT_FOLDER'], "validation_report.pdf")
        
        if not os.path.exists(pdf_report_path):
            return "PDF report not found", 404

        # Get the name of the uploaded file (you'll need to store this when uploading)
        uploaded_filename = session.get('uploaded_filename', 'unknown')
        
        # Get the base name of the original file (without extension)
        original_name = os.path.splitext(uploaded_filename)[0]
        
        # Get current date and time
        current_time = datetime.now().strftime("%d-%b-%y_%H%M")
        
        # Create a new filename with the original name and timestamp as a suffix
        new_filename = f"validation_report_{original_name}_{current_time}.pdf"

        def generate():
            with open(pdf_report_path, 'rb') as f:
                while True:
                    chunk = f.read(8192)  # Read in 8KB chunks
                    if not chunk:
                        break
                    yield chunk

        response = Response(
            stream_with_context(generate()),
            content_type='application/pdf'
        )
        response.headers['Content-Disposition'] = f'attachment; filename="{new_filename}"'
        return response

    except Exception as e:
        app.logger.error(f"Error in download_pdf: {str(e)}")
        return "Error streaming PDF", 500


@app.route('/outputs/<path:filename>')
def serve_output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    is_production = os.environ.get('ENVIRONMENT') == 'production'
    
    if is_production:
        # Production in Google App Engine
        app.run(host='0.0.0.0', port=port)
    else:
        # Local development
        app.run(debug=True)