import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pdfkit
from jinja2 import Template

def get_media_control_intercept_cols(df):
    # Define column categories
    media_cols = [col for col in df.columns if col.startswith(('paid_media_', 'owned_media_')) or 'promo' in col.lower()]
    intercept_cols = ['varying_intercept']
    uncategorized_cols = ['date', 'year', 'hierarchy']
    control_cols = [col for col in df.columns if col not in media_cols and col not in intercept_cols + uncategorized_cols]
    return media_cols, control_cols, intercept_cols, uncategorized_cols


def make_report(new_file_path, old_file_path, output_folder):
    # Load the data
    df_new = pd.read_csv(new_file_path)
    df_old = pd.read_csv(old_file_path)

    # Convert date columns to datetime format
    df_new['date'] = pd.to_datetime(df_new['date'])
    df_old['date'] = pd.to_datetime(df_old['date'])

    # Sort both dataframes by 'date' and 'hierarchy' to ensure alignment
    df_new = df_new.sort_values(by=['date', 'hierarchy']).reset_index(drop=True)
    df_old = df_old.sort_values(by=['date', 'hierarchy']).reset_index(drop=True)

    # Find common dates present in both df_old and df_new
    common_dates = pd.to_datetime(
        np.intersect1d(df_old['date'].values, df_new['date'].values)
    )

    # Filter dates until March 2024
    last_model_train_date = '2024-03-31'
    common_dates = common_dates[common_dates <= last_model_train_date]

    # Filter dataframes for common dates only
    df_new = df_new[df_new['date'].isin(common_dates)]
    df_old = df_old[df_old['date'].isin(common_dates)]

    # Identify columns missing in each dataset
    missing_in_new = set(df_old.columns) - set(df_new.columns)
    missing_in_old = set(df_new.columns) - set(df_old.columns)


    # Merge datasets on 'date' and 'hierarchy' for direct comparison
    df_combined = pd.merge(df_new, df_old, on=['date', 'hierarchy'], suffixes=('_new', '_old'))

    # Identify columns present in both datasets for comparison
    common_columns = set(df_new.columns).intersection(set(df_old.columns)) - {'date', 'hierarchy'}
    columns_to_compare = list(common_columns)


    # Calculate percentage differences for common columns only
    for col in columns_to_compare:
        df_combined[f'{col}_pct_diff'] = np.where(
            df_combined[f'{col}_old'] == 0, 
            np.nan,  # Avoid division by zero by marking as NaN
            ((df_combined[f'{col}_new'] - df_combined[f'{col}_old']) / df_combined[f'{col}_old']) * 100
        )

    # Calculate overall average percentage difference for each variable across all brands and dates
    significant_changes_summary = pd.DataFrame({
        "Variable": columns_to_compare,
        "Average_Percentage_Diff": [df_combined[f"{col}_pct_diff"].abs().mean() for col in columns_to_compare]
    })

    # Sort by Average_Percentage_Diff in descending order
    significant_changes_summary = significant_changes_summary.sort_values(
        by="Average_Percentage_Diff", 
        ascending=False
    )

    # Format the Average_Percentage_Diff column as a percentage
    significant_changes_summary["Average_Percentage_Diff"] = significant_changes_summary["Average_Percentage_Diff"].apply(lambda x: f"{x:.2f}%")

    # Reset the index for a clean display
    significant_changes_summary = significant_changes_summary.reset_index(drop=True)

    # Ensure 'charts' directory exists\
    chart_dir = os.path.join(output_folder, "charts")
    os.makedirs(chart_dir, exist_ok=True)

    # Loop through each brand and plot variables with >5% average difference
    chart_paths = []
    for brand in df_combined['hierarchy'].unique():
        brand_data = df_combined[df_combined['hierarchy'] == brand].copy()  # Use .copy() to avoid SettingWithCopyWarning
        
        for col in columns_to_compare:
            # Calculate the average percentage difference for this column and brand
            avg_pct_diff = brand_data[f'{col}_pct_diff'].abs().mean()
            
            # If the average percentage difference exceeds 5%, plot it
            if avg_pct_diff >= 5:
                plt.figure(figsize=(10, 6))
                plt.plot(brand_data['date'], brand_data[f'{col}_new'], label='New Data', marker='o', linestyle='-', alpha=0.7)
                plt.plot(brand_data['date'], brand_data[f'{col}_old'], label='Old Data', marker='x', linestyle='--', alpha=0.7)
                
                # Add a text annotation for the average percentage difference on the plot
                avg_pct_diff_text = f'Avg. % Diff: {avg_pct_diff:.2f}%'
                plt.annotate(avg_pct_diff_text, xy=(0.05, 0.95), xycoords='axes fraction',
                            fontsize=12, ha='left', va='top', color='red', fontweight='bold')

                plt.title(f'{brand} - Comparison of {col}')
                plt.xlabel('Date')
                # plt.ylabel(col)
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                
                # Save each chart as a PNG file with the brand and column in the filename
                chart_filename = f"{brand}_{col}_comparison.png"
                chart_path = os.path.join(chart_dir, chart_filename)
                plt.savefig(chart_path)
                chart_paths.append(f"outputs/charts/{chart_filename}")  # Store relative path
                plt.close()

    # print("Charts have been saved for variables with significant changes.")


    # Function to calculate the component percentages for a given DataFrame, grouped by brand
    def calculate_components_by_brand(df):
        results = []
        media_cols, control_cols, intercept_cols, uncategorized_cols = get_media_control_intercept_cols(df)
    
        for brand, brand_data in df.groupby('hierarchy'):
            media_total = brand_data[media_cols].sum().sum()
            control_total = brand_data[control_cols].sum().sum()
            intercept_total = brand_data[intercept_cols].sum().sum() if all(col in df.columns for col in intercept_cols) else 0
            grand_total = media_total + control_total + intercept_total
            
            # Calculate percentages
            media_pct = (media_total / grand_total) * 100 if grand_total else 0
            control_pct = (control_total / grand_total) * 100 if grand_total else 0
            intercept_pct = (intercept_total / grand_total) * 100 if grand_total else 0
            
            # Append results for each brand
            results.append({
                'Brand': brand,
                'Media (%)': media_pct,
                'Controls (%)': control_pct,
                'Intercept (%)': intercept_pct
            })
        
        # Sort the results by brand name
        return pd.DataFrame(results).sort_values('Brand', key=lambda x: x.str.extract('(\d+)', expand=False).astype(int))

    # Calculate brand-wise percentages for old and new data
    old_brandwise = calculate_components_by_brand(df_old).set_index('Brand')
    new_brandwise = calculate_components_by_brand(df_new).set_index('Brand')

    # Combine old and new data, and calculate the differences
    comparison = pd.DataFrame({
        'Media (%) Old': old_brandwise['Media (%)'],
        'Media (%) New': new_brandwise['Media (%)'],
        'Media Diff (%)': new_brandwise['Media (%)'] - old_brandwise['Media (%)'],
        'Controls (%) Old': old_brandwise['Controls (%)'],
        'Controls (%) New': new_brandwise['Controls (%)'],
        'Controls Diff (%)': new_brandwise['Controls (%)'] - old_brandwise['Controls (%)'],
        'Intercept (%) Old': old_brandwise['Intercept (%)'],
        'Intercept (%) New': new_brandwise['Intercept (%)'],
        'Intercept Diff (%)': new_brandwise['Intercept (%)'] - old_brandwise['Intercept (%)']
    })

    # Round for clarity
    comparison = comparison.round(2)
    comparison = comparison.reset_index()

    # Display the brand-wise comparison
    # print("Comparison of Media, Controls, and Intercept contributions:")
    # print(comparison)


    # Extract year if not already present
    df_old['year'] = df_old['date'].dt.year
    df_new['year'] = df_new['date'].dt.year

    # Function to calculate the component percentages for a given DataFrame, grouped by year
    def calculate_components_by_year(df):
        results = []
        media_cols, control_cols, intercept_cols, uncategorized_cols = get_media_control_intercept_cols(df)
    
        for year, year_data in df.groupby('year'):
            media_total = year_data[media_cols].sum().sum()
            control_total = year_data[control_cols].sum().sum()
            intercept_total = year_data[intercept_cols].sum().sum() if all(col in df.columns for col in intercept_cols) else 0
            grand_total = media_total + control_total + intercept_total
            
            # Calculate percentages
            media_pct = (media_total / grand_total) * 100 if grand_total else 0
            control_pct = (control_total / grand_total) * 100 if grand_total else 0
            intercept_pct = (intercept_total / grand_total) * 100 if grand_total else 0
            
            # Append results for each year
            results.append({
                'Year': year,
                'Media (%)': media_pct,
                'Controls (%)': control_pct,
                'Intercept (%)': intercept_pct
            })
        
        return pd.DataFrame(results)

    # Calculate year-wise percentages for old and new data
    old_yearwise = calculate_components_by_year(df_old).set_index('Year')
    new_yearwise = calculate_components_by_year(df_new).set_index('Year')

    # Combine old and new data, and calculate the differences
    yearwise_comparison = pd.DataFrame({
        'Media (%) Old': old_yearwise['Media (%)'],
        'Media (%) New': new_yearwise['Media (%)'],
        'Media Diff (%)': new_yearwise['Media (%)'] - old_yearwise['Media (%)'],
        'Controls (%) Old': old_yearwise['Controls (%)'],
        'Controls (%) New': new_yearwise['Controls (%)'],
        'Controls Diff (%)': new_yearwise['Controls (%)'] - old_yearwise['Controls (%)'],
        'Intercept (%) Old': old_yearwise['Intercept (%)'],
        'Intercept (%) New': new_yearwise['Intercept (%)'],
        'Intercept Diff (%)': new_yearwise['Intercept (%)'] - old_yearwise['Intercept (%)']
    })

    # Round for clarity
    yearwise_comparison = yearwise_comparison.round(2)
    yearwise_comparison = yearwise_comparison.reset_index()

    # Display the year-wise comparison
    # print("Year-wise MMM Component Comparison:")
    # print(yearwise_comparison)



    # Function to calculate component percentages for each brand-year combination
    def calculate_components_by_brand_year(df):
        df = df[df['year'] >= 2023]
        media_cols, control_cols, intercept_cols, uncategorized_cols = get_media_control_intercept_cols(df)
    
        results = []
        for (year, brand), brand_data in df.groupby(['year', 'hierarchy']):
            media_total = brand_data[media_cols].sum().sum()
            control_total = brand_data[control_cols].sum().sum()
            intercept_total = brand_data[intercept_cols].sum().sum() if all(col in df.columns for col in intercept_cols) else 0
            grand_total = media_total + control_total + intercept_total
            
            # Calculate percentages
            media_pct = (media_total / grand_total) * 100 if grand_total else 0
            control_pct = (control_total / grand_total) * 100 if grand_total else 0
            intercept_pct = (intercept_total / grand_total) * 100 if grand_total else 0
            
            # Append results for each brand-year combination
            results.append({
                'hierarchy': brand,
                'year': year,
                'Media (%)': media_pct,
                'Controls (%)': control_pct,
                'Intercept (%)': intercept_pct
            })
        
        return pd.DataFrame(results)

    # Calculate brand-year percentages for old and new data
    old_brand_year = calculate_components_by_brand_year(df_old).set_index(['hierarchy', 'year'])
    new_brand_year = calculate_components_by_brand_year(df_new).set_index(['hierarchy', 'year'])

    # Combine old and new data, and calculate the differences
    brand_year_comparison = pd.DataFrame({
        'Media (%) Old': old_brand_year['Media (%)'],
        'Media (%) New': new_brand_year['Media (%)'],
        'Media Diff (%)': new_brand_year['Media (%)'] - old_brand_year['Media (%)'],
        'Controls (%) Old': old_brand_year['Controls (%)'],
        'Controls (%) New': new_brand_year['Controls (%)'],
        'Controls Diff (%)': new_brand_year['Controls (%)'] - old_brand_year['Controls (%)'],
        'Intercept (%) Old': old_brand_year['Intercept (%)'],
        'Intercept (%) New': new_brand_year['Intercept (%)'],
        'Intercept Diff (%)': new_brand_year['Intercept (%)'] - old_brand_year['Intercept (%)']
    }).reset_index()

    # Custom sorting function for brand names
    def brand_sort_key(brand):
        return int(''.join(filter(str.isdigit, brand)))

    # Sort by brand (hierarchy) and year
    brand_year_comparison = brand_year_comparison.sort_values(
        by=['hierarchy', 'year'],
        key=lambda x: x.map(brand_sort_key) if x.name == 'hierarchy' else x
    ).reset_index(drop=True)

    # Round for clarity
    brand_year_comparison = brand_year_comparison.round(2)

    # Display the brand-year comparison table in the specified format
    # print("Brand-Year-wise MMM Component Comparison:")
    # print(brand_year_comparison)


    html_template="""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Data Validation Report</title>
        <style>
            /* General styling */
            body { font-family: Arial, sans-serif; margin: 60px; color: #333; line-height: 1.6; }
            h1, h2, h3 { color: #4A90E2; }
            h1 { text-align: center; }
            
            /* Table styling */
            table { width: 100%; border-collapse: collapse; margin-top: 20px; table-layout: auto; }
            th, td { padding: 12px; text-align: left; word-wrap: break-word; white-space: normal; }
            th { background-color: #f4f6f8; color: #333; }
            tr:nth-child(even) { background-color: #f9f9f9; }

            /* Section styling */
            .section { margin-bottom: 40px; }
            .section h2 { border-bottom: 2px solid #ddd; padding-bottom: 8px; }
            ul { list-style-type: disc; padding-left: 20px; }
            ul.missing-columns { list-style-type: circle; }
            
            /* Chart table styling */
            .chart-table { width: 100%; border-collapse: collapse; margin-top: 20px; }
            .chart-table td { width: 33.33%; padding: 10px; vertical-align: top; }
            img { width: 100%; height: auto; border: 1px solid #ddd; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); cursor: pointer; }
            
            /* DataTables custom styling */
            .dataTables_wrapper .dataTables_filter input { border-radius: 4px; padding: 4px; }
            .dataTables_wrapper .dataTables_paginate .paginate_button { padding: 5px; margin: 2px; }

            .collapsible {
                background-color: #f1f1f1;
                color: #444;
                cursor: pointer;
                padding: 18px;
                width: 100%;
                border: none;
                text-align: left;
                outline: none;
                font-size: 15px;
            }

            .active, .collapsible:hover {
                background-color: #ccc;
            }

            .content {
                padding: 0 18px;
                display: none;
                overflow: hidden;
                background-color: #f9f9f9;
            }
        </style>

        <!-- Include DataTables only if not generating a PDF -->
        <link rel="stylesheet" href="https://cdn.datatables.net/1.11.3/css/jquery.dataTables.min.css">
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script src="https://cdn.datatables.net/1.11.3/js/jquery.dataTables.min.js"></script>

        <!-- Lightbox CSS and JS for image popup (optional) -->
        <link href="https://cdnjs.cloudflare.com/ajax/libs/lightbox2/2.11.3/css/lightbox.min.css" rel="stylesheet">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/lightbox2/2.11.3/js/lightbox.min.js"></script>
        
        <!-- Font Awesome CSS for icons -->
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
    </head>
    <body>
        <h1>Data Validation Report</h1>

        <div class="section">
            <p>New Filename: {{ new_file_name }}</p>
            <p>Old Filename: {{ old_file_name }}</p>
            <p>Showing Data till: {{ last_model_train_date }}</p>
        </div>
        {% if missing_in_new or missing_in_old %}
            <div class="section">
                <h2>Missing Columns</h2>
                {% if missing_in_new%}
                    <h3>Columns Missing in New File</h3>
                    <ul class="missing-columns">
                        {% for col in missing_in_new %}
                            <li>{{ col }}</li>
                        {% endfor %}
                    </ul>
                {% endif %}
                {% if missing_in_old %}
                    <h3>Columns Missing in Old File</h3>
                    <ul class="missing-columns">
                        {% for col in missing_in_old %}
                            <li>{{ col }}</li>
                        {% endfor %}
                    </ul>
                {% endif %}
            </div>
        {% endif %}

        <div class="section">
            <h2>Comparison of Media, Controls, and Intercept Contributions</h2>
            {{ media_comparison | safe }}
        </div>

        <div class="section">
            <h2>Year-wise Comparison of Media, Controls, and Intercept Contributions</h2>
            {{ yearly_comparison | safe }}
        </div>

        <div class="section">
            <h2>Brand-wise and Year-wise Comparison (Excluding 2021 and 2022)</h2>
            {{ yearly_brand_comparison | safe }}
        </div>

        {% if not for_pdf %}
        <div class="section">
            <h2>Variable Categories</h2>
            
            <button type="button" class="collapsible">
                <i class="fas fa-chevron-down"></i> Media Variables
            </button>
            <div class="content">
                <ul>
                    {% for col in media_cols %}
                        <li>{{ col }}</li>
                    {% endfor %}
                </ul>
            </div>

            <button type="button" class="collapsible">
                <i class="fas fa-chevron-down"></i> Control Variables
            </button>
            <div class="content">
                <ul>
                    {% for col in control_cols %}
                        <li>{{ col }}</li>
                    {% endfor %}
                </ul>
            </div>

            <button type="button" class="collapsible">
                <i class="fas fa-chevron-down"></i> Intercept Variables
            </button>
            <div class="content">
                <ul>
                    {% for col in intercept_cols %}
                        <li>{{ col }}</li>
                    {% endfor %}
                </ul>
            </div>

            <button type="button" class="collapsible">
                <i class="fas fa-chevron-down"></i> Uncategorized Variables
            </button>
            <div class="content">
                <ul>
                    {% for col in uncategorized_cols %}
                        <li>{{ col }}</li>
                    {% endfor %}
                </ul>
            </div>
        </div>
        {% endif %}
        
        <div class="section">
            <h2>Summary of Significant Changes</h2>
            {{ summary_table | safe }}
        </div>
        
        <div class="section">
            <h2>Brand-wise Charts for Variables with >5% Difference</h2>
            {% if not for_pdf %}
                <p><b>NOTE: </b>Click on image to increase its size</p>
            {% endif %}
            <table class="chart-table">
                <tr>
                {% for chart in brand_wise_charts %}
                    <td>
                        {% if not for_pdf %}
                            <a href="{{ chart }}" data-lightbox="charts">
                                <img src="{{ chart }}" alt="Chart">
                            </a>
                        {% else %}
                            <img src="{{ chart }}" alt="Chart">
                        {% endif %}
                    </td>
                    {% if loop.index % 3 == 0 %}</tr><tr>{% endif %}
                {% endfor %}
                </tr>
            </table>
        </div>
                
        <!-- Initialize DataTables only if not generating PDF -->
        {% if not for_pdf %}
        <script>
            $(document).ready(function() {
                $('#media-comparison').DataTable();
                $('#yearly-comparison').DataTable();
                $('#yearly-brand-comparison').DataTable();
                $('#summary-table').DataTable();
            });

            var coll = document.getElementsByClassName("collapsible");
            var i;

            for (i = 0; i < coll.length; i++) {
                coll[i].addEventListener("click", function() {
                    this.classList.toggle("active");
                    var content = this.nextElementSibling;
                    if (content.style.display === "block") {
                        content.style.display = "none";
                    } else {
                        content.style.display = "block";
                    }
                });
            }
        </script>
        {% endif %}
    </body>
    </html>

    """


    # Extract headers and body for each DataFrame to avoid duplication
    media_comparison_html = comparison.to_html(index=False, classes="display", table_id="media-comparison")
    yearly_comparison_html = yearwise_comparison.to_html(index=False, classes="display", table_id="yearly-comparison")
    yearly_brand_comparison_html = brand_year_comparison.to_html(index=False, classes="display", table_id="yearly-brand-comparison")
    summary_table_html = significant_changes_summary.to_html(index=False, classes="display", table_id="summary-table")

    current_dir = os.getcwd()
    # Define column categories
    media_cols, control_cols, intercept_cols, uncategorized_cols = get_media_control_intercept_cols(df_new)
    

    # Render the HTML with the Jinja2 template
    html_content = Template(html_template).render(
        missing_in_new=missing_in_new,
        missing_in_old=missing_in_old,
        media_comparison=media_comparison_html,
        yearly_comparison=yearly_comparison_html,
        yearly_brand_comparison=yearly_brand_comparison_html,
        summary_table=summary_table_html,
        brand_wise_charts=chart_paths,
        current_dir = "",
        new_file_name = new_file_path,
        old_file_name = old_file_path,
        last_model_train_date = last_model_train_date,
        media_cols=media_cols,
        control_cols=control_cols,
        intercept_cols=intercept_cols,
        uncategorized_cols = uncategorized_cols,
        for_pdf = False,
    )

    html_report_path = os.path.join(output_folder, "validation_report.html")
    pdf_report_path = os.path.join(output_folder, "validation_report.pdf")

    # Save HTML report
    with open(html_report_path, "w") as f:
        f.write(html_content)


    # Render the HTML with the Jinja2 template
    pdf_content = Template(html_template).render(
        missing_in_new=missing_in_new,
        missing_in_old=missing_in_old,
        media_comparison=media_comparison_html,
        yearly_comparison=yearly_comparison_html,
        yearly_brand_comparison=yearly_brand_comparison_html,
        summary_table=summary_table_html,
        brand_wise_charts=[os.path.join(current_dir, path) for path in chart_paths],
        current_dir = current_dir,
        new_file_name = new_file_path,
        old_file_name = old_file_path,
        last_model_train_date = last_model_train_date,
        for_pdf=True  # Pass the flag to the template
    )

    # Convert HTML to PDF with pdfkit
    pdfkit.from_string(pdf_content, pdf_report_path, options={
        'enable-local-file-access': None,
        'quiet': ''
    })

    return html_report_path, pdf_report_path