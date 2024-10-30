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
from scipy.stats import pearsonr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ColumnCategories:
    """Groups columns into media, control, intercept and other categories"""
    media_cols: List[str]
    control_cols: List[str]
    intercept_cols: List[str]
    other_cols: List[str]

def get_media_control_intercept_cols(df: pd.DataFrame) -> ColumnCategories:
    """Categorizes DataFrame columns into media, control, intercept and other categories"""
    media_cols = [col for col in df.columns if col.startswith(('paid_media_', 'owned_media_')) or 'promo' in col.lower()]
    intercept_cols = ['varying_intercept']
    base_cols = ['date', 'year', 'hierarchy']
    control_cols = [col for col in df.columns if col not in media_cols + intercept_cols + base_cols]
    
    return ColumnCategories(
        media_cols=media_cols,
        control_cols=control_cols,
        intercept_cols=intercept_cols,
        other_cols=base_cols
    )

def load_and_prepare_data(
    new_file_path: str,
    old_file_path: str,
    country_name: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads, filters and prepares data from new and old model files"""
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

        if end_date:
            common_dates = common_dates[common_dates <= end_date]
        
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
    """Identifies columns that exist in one DataFrame but not in the other"""
    missing_in_new = set(df_old.columns) - set(df_new.columns)
    missing_in_old = set(df_new.columns) - set(df_old.columns)
    return missing_in_new, missing_in_old

def create_combined_dataframe(
    df_new: pd.DataFrame,
    df_old: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str]]:
    """Merges new and old DataFrames and identifies common columns for comparison"""
    df_combined = pd.merge(
        df_new,
        df_old,
        on=['date', 'hierarchy'],
        suffixes=('_new', '_old')
    )
    common_columns = set(df_new.columns).intersection(set(df_old.columns)) - {'date', 'year', 'hierarchy'}
    return df_combined, list(common_columns)

def calculate_percentage_differences(
    df_combined: pd.DataFrame,
    columns_to_compare: List[str]
) -> pd.DataFrame:
    """Calculates percentage differences between new and old values for each column"""
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
    """Generates summary statistics of differences between new and old values"""
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
    """Creates comparison chart for a specific brand and column with sales correlation"""
    try:
        # Calculate total sales for new and old data
        col_categories = get_media_control_intercept_cols(brand_data)
        all_contrib_cols = (
            col_categories.media_cols + 
            col_categories.control_cols + 
            col_categories.intercept_cols
        )
        
        # Calculate totals removing '_new' and '_old' suffixes from columns
        contrib_cols_new = [c for c in brand_data.columns if c.endswith('_new') 
                          and c[:-4] in all_contrib_cols]
        contrib_cols_old = [c for c in brand_data.columns if c.endswith('_old') 
                          and c[:-4] in all_contrib_cols]
        
        total_sales_new = brand_data[contrib_cols_new].sum(axis=1)
        total_sales_old = brand_data[contrib_cols_old].sum(axis=1)
        
        # Calculate correlations using scipy's pearsonr
        corr_new, p_new = pearsonr(brand_data[f'{col}_new'], total_sales_new)
        corr_old, p_old = pearsonr(brand_data[f'{col}_old'], total_sales_old)

        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(brand_data['date'], brand_data[f'{col}_new'],
                label='New Data', marker='o', linestyle='-', alpha=0.7)
        plt.plot(brand_data['date'], brand_data[f'{col}_old'],
                label='Old Data', marker='x', linestyle='--', alpha=0.7)
        
        # Add annotations with p-values
        plt.annotate(
            f'Avg. % Diff: {avg_pct_diff:.2f}%\n'
            f'Corr. with Sales (New): {corr_new:.2f} (p={p_new:.3f})\n'
            f'Corr. with Sales (Old): {corr_old:.2f} (p={p_old:.3f})',
            xy=(0.05, 0.95),
            xycoords='axes fraction',
            fontsize=10,
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
        
        chart_filename = f"{brand}_{col}.png"
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
    """Generates comparison charts for columns with significant differences"""
    try:
        chart_dir = os.path.join(output_folder, "charts")
        os.makedirs(chart_dir, exist_ok=True)
        chart_paths = []
        
        # Calculate max diff for each brand-column combination
        diff_data = []
        for brand in df_combined['hierarchy'].unique():
            brand_data = df_combined[df_combined['hierarchy'] == brand].copy()
            
            for col in columns_to_compare:
                avg_pct_diff = brand_data[f'{col}_pct_diff'].abs().mean()
                if avg_pct_diff >= 5:
                    diff_data.append({
                        'brand': brand,
                        'column': col,
                        'diff': avg_pct_diff,
                        'data': brand_data
                    })
            
            del brand_data
            gc.collect()
            
        # Sort by diff descending and generate charts
        diff_data.sort(key=lambda x: x['diff'], reverse=True)
        
        for item in diff_data:
            chart_path = plot_comparison_chart(
                item['data'], 
                item['brand'],
                item['column'],
                item['diff'],
                chart_dir
            )
            chart_paths.append(f"outputs/charts/{os.path.basename(chart_path)}")
            
            del item['data']
            gc.collect()
        
        return chart_paths
    except Exception as e:
        logger.error(f"Error in generate_comparison_charts: {str(e)}")
        raise

def calculate_components_by_brand(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates percentage contribution of each component by brand"""
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
        
        return pd.DataFrame(results).set_index('Brand')
    except Exception as e:
        logger.error(f"Error in calculate_components_by_brand: {str(e)}")
        raise

def calculate_components_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates percentage contribution of each component by year"""
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
    """Calculates percentage contribution of each component by brand and year"""
    try:
        df = df[df['year'] >= 2023]  # Filter for recent years
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
                'Brand': brand,
                'Year': year,
                'Media (%)': (media_total / grand_total * 100) if grand_total else 0,
                'Controls (%)': (control_total / grand_total * 100) if grand_total else 0,
                'Intercept (%)': (intercept_total / grand_total * 100) if grand_total else 0
            })
        
        return pd.DataFrame(results).set_index(['Brand', 'Year'])
    
    except Exception as e:
        logger.error(f"Error in calculate_components_by_brand_year: {str(e)}")
        raise


def get_html_template() -> Template:
    """Loads HTML template from file"""
    template_path = os.path.join(os.path.dirname(__file__), 'templates', 'report_template.html')
    with open(template_path, 'r', encoding='utf-8') as f:
        return Template(f.read())

def generate_report(template_data: Dict[str, Any], output_folder: str) -> Tuple[str, str]:
    """Generates HTML and PDF validation reports using template data"""
    try:
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Get HTML template
        template = get_html_template()
        
        # Generate HTML report with for_pdf=False
        html_content = template.render(
            **template_data,
            for_pdf=False
        )
        
        html_report_path = os.path.join(output_folder, "validation_report.html")
        with open(html_report_path, "w", encoding='utf-8') as f:
            f.write(html_content)
        
        # Update the chart paths for PDF
        template_data['brand_wise_charts'] = [
            os.path.join(os.getcwd(), path) 
            for path in template_data['brand_wise_charts']
        ]

        # Generate PDF report with for_pdf=True and absolute paths
        pdf_content = template.render(
            **template_data,
            for_pdf=True
        )
        
        # Generate PDF report using the wrapper script
        pdf_report_path = os.path.join(output_folder, "validation_report.pdf")
        
        pdfkit.from_string(pdf_content, pdf_report_path, options={
            'enable-local-file-access': None,
            'quiet': ''
        })
        
        logger.info(f"Reports generated successfully at {output_folder}")
        return html_report_path, pdf_report_path
        
    except Exception as e:
        logger.error(f"Error in generate_report: {str(e)}")
        raise

def make_validation_report(
    new_file_path: str,
    old_file_path: str,
    output_folder: str,
    country_name: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Tuple[str, str]:
    """Main function that orchestrates the validation report generation process"""
    try:
        logger.info("Starting validation report generation...")

        # Load and prepare data
        df_new, df_old = load_and_prepare_data(
            new_file_path = new_file_path,
            old_file_path = old_file_path,
            country_name = country_name,
            start_date = start_date,
            end_date = end_date,
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
        
        # Get column categories
        col_categories = get_media_control_intercept_cols(df_new)

        # Calculate components
        # Calculate brand-wise percentages
        old_brandwise = calculate_components_by_brand(df_old)
        new_brandwise = calculate_components_by_brand(df_new)
        
        # Create comparison DataFrame
        brand_components = pd.DataFrame({
            'Media (%) Old': old_brandwise['Media (%)'],
            'Media (%) New': new_brandwise['Media (%)'],
            'Media Diff (%)': new_brandwise['Media (%)'] - old_brandwise['Media (%)'],
            'Controls (%) Old': old_brandwise['Controls (%)'],
            'Controls (%) New': new_brandwise['Controls (%)'],
            'Controls Diff (%)': new_brandwise['Controls (%)'] - old_brandwise['Controls (%)'],
            'Intercept (%) Old': old_brandwise['Intercept (%)'],
            'Intercept (%) New': new_brandwise['Intercept (%)'],
            'Intercept Diff (%)': new_brandwise['Intercept (%)'] - old_brandwise['Intercept (%)']
        }).round(2).reset_index()

         # Calculate year-wise percentages
        old_yearwise = calculate_components_by_year(df_old).set_index('Year')
        new_yearwise = calculate_components_by_year(df_new).set_index('Year')
        
        # Create year-wise comparison
        year_components = pd.DataFrame({
            'Media (%) Old': old_yearwise['Media (%)'],
            'Media (%) New': new_yearwise['Media (%)'],
            'Media Diff (%)': new_yearwise['Media (%)'] - old_yearwise['Media (%)'],
            'Controls (%) Old': old_yearwise['Controls (%)'],
            'Controls (%) New': new_yearwise['Controls (%)'],
            'Controls Diff (%)': new_yearwise['Controls (%)'] - old_yearwise['Controls (%)'],
            'Intercept (%) Old': old_yearwise['Intercept (%)'],
            'Intercept (%) New': new_yearwise['Intercept (%)'],
            'Intercept Diff (%)': new_yearwise['Intercept (%)'] - old_yearwise['Intercept (%)']
        }).round(2).reset_index()

        # Calculate brand-year percentages
        old_brand_year = calculate_components_by_brand_year(df_old)
        new_brand_year = calculate_components_by_brand_year(df_new)
        
        # Create brand-year comparison
        brand_year_components = pd.DataFrame({
            'Media (%) Old': old_brand_year['Media (%)'],
            'Media (%) New': new_brand_year['Media (%)'],
            'Media Diff (%)': new_brand_year['Media (%)'] - old_brand_year['Media (%)'],
            'Controls (%) Old': old_brand_year['Controls (%)'],
            'Controls (%) New': new_brand_year['Controls (%)'],
            'Controls Diff (%)': new_brand_year['Controls (%)'] - old_brand_year['Controls (%)'],
            'Intercept (%) Old': old_brand_year['Intercept (%)'],
            'Intercept (%) New': new_brand_year['Intercept (%)'],
            'Intercept Diff (%)': new_brand_year['Intercept (%)'] - old_brand_year['Intercept (%)']
        }).round(2).reset_index()
        
        # Prepare template data
        template_data = {
            'new_file_name': os.path.basename(new_file_path),
            'old_file_name': os.path.basename(old_file_path),
            'end_date': end_date,
            'missing_in_new': missing_in_new,
            'missing_in_old': missing_in_old,
            'media_comparison': brand_components.to_html(index=False, classes="display", table_id="media-comparison"),
            'yearly_comparison': year_components.to_html(index=False, classes="display", table_id="yearly-comparison"),
            'yearly_brand_comparison': brand_year_components.to_html(index=False, classes="display", table_id="yearly-brand-comparison"),
            'summary_table': summary_table.to_html(index=False, classes="display", table_id="summary-table"),
            'brand_wise_charts': chart_paths,
            'media_cols': col_categories.media_cols,
            'control_cols': col_categories.control_cols,
            'intercept_cols': col_categories.intercept_cols,
            'other_cols': col_categories.other_cols  # Updated to match renamed field
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
#             end_date="2024-03-31"
#         )
#         print(f"Reports generated successfully:\nHTML: {html_path}\nPDF: {pdf_path}")
#     except Exception as e:
#         print(f"Error generating reports: {str(e)}")