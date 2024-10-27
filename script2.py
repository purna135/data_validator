import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pdfkit
from jinja2 import Template
import gc
from typing import Tuple, List, Set, Dict, Any
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ColumnCategories:
    """Data class to store column categories"""
    media_cols: List[str]
    control_cols: List[str]
    intercept_cols: List[str]
    uncategorized_cols: List[str]

def get_media_control_intercept_cols(df: pd.DataFrame) -> ColumnCategories:
    """Categorize columns into media, control, intercept, and uncategorized."""
    media_cols = [col for col in df.columns if col.startswith(('paid_media_', 'owned_media_')) or 'promo' in col.lower()]
    intercept_cols = ['varying_intercept']
    uncategorized_cols = ['date', 'year', 'hierarchy']
    control_cols = [col for col in df.columns if col not in media_cols + intercept_cols + uncategorized_cols]
    
    return ColumnCategories(
        media_cols=media_cols,
        control_cols=control_cols,
        intercept_cols=intercept_cols,
        uncategorized_cols=uncategorized_cols
    )

def load_and_prepare_data(
    new_file_path: str,
    old_file_path: str,
    last_model_train_date: str = '2024-03-31'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and prepare dataframes with common dates."""
    try:
        logger.info("Loading data files...")
        df_new = pd.concat(pd.read_csv(new_file_path, chunksize=10000))
        df_old = pd.concat(pd.read_csv(old_file_path, chunksize=10000))
        
        # Convert and filter dates
        for df in [df_new, df_old]:
            df['date'] = pd.to_datetime(df['date'])
            df.sort_values(by=['date', 'hierarchy'], inplace=True)
            df.reset_index(drop=True, inplace=True)
        
        common_dates = pd.to_datetime(
            np.intersect1d(df_old['date'].values, df_new['date'].values)
        )
        common_dates = common_dates[common_dates <= last_model_train_date]
        
        # Filter dataframes
        df_new = df_new[df_new['date'].isin(common_dates)]
        df_old = df_old[df_old['date'].isin(common_dates)]
        df_old['year'] = df_old['date'].dt.year
        df_new['year'] = df_new['date'].dt.year

        logger.info("Data loading and preparation completed successfully")
        return df_new, df_old
        
    except Exception as e:
        logger.error(f"Error in load_and_prepare_data: {str(e)}")
        raise

def identify_missing_columns(
    df_new: pd.DataFrame,
    df_old: pd.DataFrame
) -> Tuple[Set[str], Set[str]]:
    """Identify columns missing in each dataset."""
    missing_in_new = set(df_old.columns) - set(df_new.columns)
    missing_in_old = set(df_new.columns) - set(df_old.columns)
    return missing_in_new, missing_in_old

def create_combined_dataframe(
    df_new: pd.DataFrame,
    df_old: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str]]:
    """Create combined dataframe for comparison."""
    df_combined = pd.merge(
        df_new,
        df_old,
        on=['date', 'hierarchy'],
        suffixes=('_new', '_old')
    )
    common_columns = set(df_new.columns).intersection(set(df_old.columns)) - {'date', 'hierarchy'}
    return df_combined, list(common_columns)

def calculate_percentage_differences(
    df_combined: pd.DataFrame,
    columns_to_compare: List[str]
) -> pd.DataFrame:
    """Calculate percentage differences for common columns."""
    for col in columns_to_compare:
        df_combined[f'{col}_pct_diff'] = np.where(
            df_combined[f'{col}_old'] == 0,
            np.nan,
            ((df_combined[f'{col}_new'] - df_combined[f'{col}_old']) / df_combined[f'{col}_old']) * 100
        )
    return df_combined


def generate_summary_table(
    df_combined: pd.DataFrame,
    columns_to_compare: List[str]
) -> pd.DataFrame:
    """Generate summary of significant changes."""
    try:
        summary = pd.DataFrame({
            "Variable": columns_to_compare,
            "Average_Percentage_Diff": [
                df_combined[f"{col}_pct_diff"].abs().mean() 
                for col in columns_to_compare
            ]
        })
        summary = summary.sort_values("Average_Percentage_Diff", ascending=False)
        summary["Average_Percentage_Diff"] = summary["Average_Percentage_Diff"].apply(
            lambda x: f"{x:.2f}%"
        )
        return summary.reset_index(drop=True)
    except Exception as e:
        logger.error(f"Error in generate_summary_table: {str(e)}")
        raise

def plot_comparison_chart(
    brand_data: pd.DataFrame,
    brand: str,
    col: str,
    avg_pct_diff: float,
    chart_dir: str
) -> str:
    """Plot individual comparison chart."""
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(brand_data['date'], brand_data[f'{col}_new'],
                label='New Data', marker='o', linestyle='-', alpha=0.7)
        plt.plot(brand_data['date'], brand_data[f'{col}_old'],
                label='Old Data', marker='x', linestyle='--', alpha=0.7)
        
        plt.annotate(
            f'Avg. % Diff: {avg_pct_diff:.2f}%',
            xy=(0.05, 0.95),
            xycoords='axes fraction',
            fontsize=12,
            ha='left',
            va='top',
            color='red',
            fontweight='bold'
        )

        plt.title(f'{brand} - Comparison of {col}')
        plt.xlabel('Date')
        plt.ylabel(col)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        chart_filename = f"{brand}_{col}_comparison.png"
        chart_path = os.path.join(chart_dir, chart_filename)
        plt.savefig(chart_path)
        plt.close('all')
        return chart_path
    except Exception as e:
        logger.error(f"Error in plot_comparison_chart: {str(e)}")
        raise
    finally:
        plt.close('all')

def generate_comparison_charts(
    df_combined: pd.DataFrame,
    columns_to_compare: List[str],
    output_folder: str
) -> List[str]:
    """Generate comparison charts for significant changes."""
    try:
        chart_dir = os.path.join(output_folder, "charts")
        os.makedirs(chart_dir, exist_ok=True)
        chart_paths = []
        
        for brand in df_combined['hierarchy'].unique():
            brand_data = df_combined[df_combined['hierarchy'] == brand].copy()
            
            for col in columns_to_compare:
                avg_pct_diff = brand_data[f'{col}_pct_diff'].abs().mean()
                
                if avg_pct_diff >= 5:
                    chart_path = plot_comparison_chart(
                        brand_data, brand, col, avg_pct_diff, chart_dir
                    )
                    chart_paths.append(f"outputs/charts/{os.path.basename(chart_path)}")
                    
            del brand_data
            gc.collect()
        
        return chart_paths
    except Exception as e:
        logger.error(f"Error in generate_comparison_charts: {str(e)}")
        raise

def calculate_components_by_brand(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate component percentages for each brand."""
    try:
        results = []
        col_categories = get_media_control_intercept_cols(df)
        
        for brand, brand_data in df.groupby('hierarchy'):
            media_total = brand_data[col_categories.media_cols].sum().sum()
            control_total = brand_data[col_categories.control_cols].sum().sum()
            intercept_total = brand_data[col_categories.intercept_cols].sum().sum() if all(
                col in df.columns for col in col_categories.intercept_cols
            ) else 0
            grand_total = media_total + control_total + intercept_total
            
            results.append({
                'Brand': brand,
                'Media (%)': (media_total / grand_total * 100) if grand_total else 0,
                'Controls (%)': (control_total / grand_total * 100) if grand_total else 0,
                'Intercept (%)': (intercept_total / grand_total * 100) if grand_total else 0
            })
        
        return pd.DataFrame(results).sort_values(
            'Brand',
            key=lambda x: x.str.extract(r'(\d+)', expand=False).astype(int)
        )
    except Exception as e:
        logger.error(f"Error in calculate_components_by_brand: {str(e)}")
        raise

def calculate_components_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate component percentages for each year."""
    try:
        results = []
        col_categories = get_media_control_intercept_cols(df)
        
        for year, year_data in df.groupby('year'):
            media_total = year_data[col_categories.media_cols].sum().sum()
            control_total = year_data[col_categories.control_cols].sum().sum()
            intercept_total = year_data[col_categories.intercept_cols].sum().sum() if all(
                col in df.columns for col in col_categories.intercept_cols
            ) else 0
            grand_total = media_total + control_total + intercept_total
            
            results.append({
                'Year': year,
                'Media (%)': (media_total / grand_total * 100) if grand_total else 0,
                'Controls (%)': (control_total / grand_total * 100) if grand_total else 0,
                'Intercept (%)': (intercept_total / grand_total * 100) if grand_total else 0
            })
        
        return pd.DataFrame(results)
    except Exception as e:
        logger.error(f"Error in calculate_components_by_year: {str(e)}")
        raise


def calculate_components_by_brand_year(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate component percentages for each brand-year combination."""
    try:
        df = df[df['year'] >= 2023]
        results = []
        col_categories = get_media_control_intercept_cols(df)
        
        for (year, brand), brand_data in df.groupby(['year', 'hierarchy']):
            media_total = brand_data[col_categories.media_cols].sum().sum()
            control_total = brand_data[col_categories.control_cols].sum().sum()
            intercept_total = brand_data[col_categories.intercept_cols].sum().sum() if all(
                col in df.columns for col in col_categories.intercept_cols
            ) else 0
            grand_total = media_total + control_total + intercept_total
            
            results.append({
                'hierarchy': brand,
                'year': year,
                'Media (%)': (media_total / grand_total * 100) if grand_total else 0,
                'Controls (%)': (control_total / grand_total * 100) if grand_total else 0,
                'Intercept (%)': (intercept_total / grand_total * 100) if grand_total else 0
            })
        
        return pd.DataFrame(results)
    except Exception as e:
        logger.error(f"Error in calculate_components_by_brand_year: {str(e)}")
        raise


def generate_html_template() -> str:
    """Generate HTML template for the report."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 1200px; margin: auto; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .chart-container { margin: 20px 0; }
            .chart { max-width: 100%; height: auto; }
            .section { margin: 30px 0; }
            h1, h2 { color: #333; }
            .alert { color: #721c24; background-color: #f8d7da; padding: 10px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Data Validation Report</h1>
            
            <div class="section">
                <h2>Files Compared</h2>
                <p>New File: {{ new_file_name }}</p>
                <p>Old File: {{ old_file_name }}</p>
            </div>

            {% if missing_in_new or missing_in_old %}
            <div class="section">
                <h2>Missing Columns</h2>
                {% if missing_in_new %}
                <div class="alert">
                    <h3>Columns missing in new dataset:</h3>
                    <ul>
                    {% for col in missing_in_new %}
                        <li>{{ col }}</li>
                    {% endfor %}
                    </ul>
                </div>
                {% endif %}

                {% if missing_in_old %}
                <div class="alert">
                    <h3>Columns missing in old dataset:</h3>
                    <ul>
                    {% for col in missing_in_old %}
                        <li>{{ col }}</li>
                    {% endfor %}
                    </ul>
                </div>
                {% endif %}
            </div>
            {% endif %}

            <div class="section">
                <h2>Summary of Changes</h2>
                {{ summary_table | safe }}
            </div>

            <div class="section">
                <h2>Brand-wise Comparisons</h2>
                {% for chart in brand_wise_charts %}
                <div class="chart-container">
                    <img class="chart" src="{{ chart }}" alt="Comparison Chart">
                </div>
                {% endfor %}
            </div>
        </div>
    </body>
    </html>
    """

def generate_report(template_data: Dict[str, Any], output_folder: str) -> Tuple[str, str]:
    """Generate HTML and PDF reports."""
    try:
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Get HTML template
        template = Template(generate_html_template())
        
        # Generate HTML report
        html_content = template.render(**template_data)
        html_report_path = os.path.join(output_folder, "validation_report.html")
        with open(html_report_path, "w", encoding='utf-8') as f:
            f.write(html_content)
        
        # Generate PDF report using the wrapper script
        pdf_report_path = os.path.join(output_folder, "validation_report.pdf")
        config = pdfkit.configuration(wkhtmltopdf=os.environ.get('WKHTMLTOPDF_CMD', 'wkhtmltopdf'))
        
        pdfkit.from_string(
            html_content,
            pdf_report_path,
            configuration=config,
            options={
                'enable-local-file-access': None,
                'quiet': '',
                'encoding': 'UTF-8'
            }
        )
        
        logger.info(f"Reports generated successfully at {output_folder}")
        return html_report_path, pdf_report_path
        
    except Exception as e:
        logger.error(f"Error in generate_report: {str(e)}")
        raise
    
def make_validation_report(
    new_file_path: str,
    old_file_path: str,
    output_folder: str,
    last_model_train_date: str = '2024-03-31'
) -> Tuple[str, str]:
    """Main function to generate validation report."""
    try:
        logger.info("Starting validation report generation...")
        
        # Load and prepare data
        df_new, df_old = load_and_prepare_data(
            new_file_path,
            old_file_path,
            last_model_train_date
        )
        
        # Identify missing columns
        missing_in_new, missing_in_old = identify_missing_columns(df_new, df_old)
        
        # Create combined dataframe
        df_combined, columns_to_compare = create_combined_dataframe(df_new, df_old)
        
        # Calculate differences
        df_combined = calculate_percentage_differences(df_combined, columns_to_compare)
        
        # Generate summary table
        summary_table = generate_summary_table(df_combined, columns_to_compare)
        
        # Generate comparison charts
        chart_paths = generate_comparison_charts(df_combined, columns_to_compare, output_folder)
        
        # Calculate components
        brand_components = calculate_components_by_brand(df_new)
        year_components = calculate_components_by_year(df_new)
        brand_year_components = calculate_components_by_brand_year(df_new)
        
        # Prepare template data
        template_data = {
            'new_file_name': os.path.basename(new_file_path),
            'old_file_name': os.path.basename(old_file_path),
            'missing_in_new': missing_in_new,
            'missing_in_old': missing_in_old,
            'summary_table': summary_table.to_html(index=False),
            'brand_wise_charts': chart_paths,
            'brand_components': brand_components.to_html(index=False),
            'year_components': year_components.to_html(index=False),
            'brand_year_components': brand_year_components.to_html(index=False)
        }
        
        # Generate reports
        html_path, pdf_path = generate_report(template_data, output_folder)
        
        logger.info("Validation report generation completed successfully")
        return html_path, pdf_path
        
    except Exception as e:
        logger.error(f"Error in make_validation_report: {str(e)}")
        raise
    finally:
        # Cleanup
        plt.close('all')
        gc.collect()

# if __name__ == "__main__":
#     # Example usage
#     try:
#         html_path, pdf_path = make_validation_report(
#             new_file_path="path/to/new/data.csv",
#             old_file_path="path/to/old/data.csv",
#             output_folder="path/to/output",
#             last_model_train_date="2024-03-31"
#         )
#         print(f"Reports generated successfully:\nHTML: {html_path}\nPDF: {pdf_path}")
#     except Exception as e:
#         print(f"Error generating reports: {str(e)}")