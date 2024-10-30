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

        # Filter by country if provided
        if country_name:
            df_new = df_new[df_new['country_name'] == country_name]
            df_old = df_old[df_old['country_name'] == country_name]

        for df in [df_new, df_old]:
            # Rename columns
            df.rename(columns={
                'dt_week': 'date',
                'brand_id': 'hierarchy',
                'Year': 'year'
            }, inplace=True)
            
            # Convert dates
            df['date'] = pd.to_datetime(df['date'], dayfirst=True)
            df.sort_values(by=['date', 'hierarchy'], inplace=True)
            df.reset_index(drop=True, inplace=True)
            
        # Get common dates between filtered datasets
        common_dates = pd.to_datetime(
            np.intersect1d(df_old['date'].values, df_new['date'].values)
        )

        # Filter by date range if provided
        if start_date:
            common_dates = common_dates[common_dates >= start_date]

        if end_date:
            common_dates = common_dates[common_dates <= end_date]
        
        
        # Filter to common dates
        df_new = df_new[df_new['date'].isin(common_dates)]
        df_old = df_old[df_old['date'].isin(common_dates)]
        
        # Add year column if not already present
        if 'year' not in df_old.columns:
            df_old['year'] = df_old['date'].dt.year
        if 'year' not in df_new.columns:
            df_new['year'] = df_new['date'].dt.year

        print("-------------------------------------------")
        print(f"Date range: {df_new['date'].min()} to {df_new['date'].max()}")
        print(f"Countries: {', '.join(df_new['country_name'].unique())}")
        print("-------------------------------------------")
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
    base_columns = ['country_name', "country_code", "hierarchy", "date", "year"]
    df_combined = pd.merge(
        df_new,
        df_old,
        on=base_columns,
        suffixes=('_new', '_old')
    )
    common_columns = set(df_new.columns).intersection(set(df_old.columns)) - set(base_columns)
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
        
        # Remove rows with 0 or NaN values
        summary = summary[
            (summary["Average_Percentage_Diff"] != 0) & 
            (~summary["Average_Percentage_Diff"].isna())
        ]
        
        # Sort by absolute difference descending
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
    """Creates comparison chart for a specific brand and column"""
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


@dataclass
class MediaCategories:
    """Groups columns into media, control, intercept and other categories"""
    paid_media_spend_cols: List[str]
    promo_spend_cols: List[str]
    retail_spend_cols: List[str]
    total_sales_cols: List[str]

def get_media_cols(df: pd.DataFrame) -> MediaCategories:
    """Categorizes DataFrame columns into media, control, intercept and other categories"""
    paid_media_spend_cols = [col for col in df.columns if col.startswith('paid_media_') and not col.startswith('paid_media_retail_')]
    promo_spend_cols = [col for col in df.columns if col.startswith('bottler_promo_')]
    retail_spend_cols = [col for col in df.columns if col.startswith('paid_media_retail_')]
    total_sales_cols = ['outcome_metric_sell_out_unit_cases']
    
    return MediaCategories(
        paid_media_spend_cols=paid_media_spend_cols,
        promo_spend_cols=promo_spend_cols,
        retail_spend_cols=retail_spend_cols,
        total_sales_cols=total_sales_cols
    )

def calculate_media_spend_by_brand(df: pd.DataFrame) -> pd.DataFrame:
    try:
        results = []
        col_categories = get_media_cols(df)
        
        for brand, brand_data in df.groupby('hierarchy'):
            paid_media_spend_total = brand_data[col_categories.paid_media_spend_cols].sum().sum()
            promo_spend_total = brand_data[col_categories.promo_spend_cols].sum().sum()
            retail_spend_total = brand_data[col_categories.retail_spend_cols].sum().sum()
            total_spend = paid_media_spend_total + promo_spend_total + retail_spend_total
            total_sales = brand_data[col_categories.total_sales_cols].sum().sum()
            
            results.append({
                'Brand': brand,
                'Paid Media (%)': (paid_media_spend_total / total_spend * 100) if total_spend else 0,
                'Promo (%)': (promo_spend_total / total_spend * 100) if total_spend else 0,
                'Retail (%)': (retail_spend_total / total_spend * 100) if total_spend else 0,
                'Total Spend': total_spend,
                'Total Sales': total_sales,
            })
        
        return pd.DataFrame(results).set_index('Brand')
    except Exception as e:
        logger.error(f"Error in calculate_media_by_brand: {str(e)}")
        raise

def calculate_media_spend_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates percentage contribution of each media type by year"""
    try:
        results = []
        col_categories = get_media_cols(df)
        
        for year, year_data in df.groupby('year'):
            paid_media_spend_total = year_data[col_categories.paid_media_spend_cols].sum().sum()
            promo_spend_total = year_data[col_categories.promo_spend_cols].sum().sum()
            retail_spend_total = year_data[col_categories.retail_spend_cols].sum().sum()
            total_spend = paid_media_spend_total + promo_spend_total + retail_spend_total
            total_sales = year_data[col_categories.total_sales_cols].sum().sum()
            
            results.append({
                'Year': year,
                'Paid Media (%)': (paid_media_spend_total / total_spend * 100) if total_spend else 0,
                'Promo (%)': (promo_spend_total / total_spend * 100) if total_spend else 0,
                'Retail (%)': (retail_spend_total / total_spend * 100) if total_spend else 0,
                'Total Spend': total_spend,
                'Total Sales': total_sales,
            })
        
        return pd.DataFrame(results)
    except Exception as e:
        logger.error(f"Error in calculate_components_by_year: {str(e)}")
        raise

def calculate_media_spend_by_brand_year(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates media spend contribution by brand and year"""
    try:
        df = df[df['year'] >= 2023]  # Filter for recent years
        results = []
        col_categories = get_media_cols(df)
        
        for (year, brand), brand_data in df.groupby(['year', 'hierarchy']):
            paid_media_spend_total = brand_data[col_categories.paid_media_spend_cols].sum().sum()
            promo_spend_total = brand_data[col_categories.promo_spend_cols].sum().sum()
            retail_spend_total = brand_data[col_categories.retail_spend_cols].sum().sum()
            total_spend = paid_media_spend_total + promo_spend_total + retail_spend_total
            total_sales = brand_data[col_categories.total_sales_cols].sum().sum()
            
            results.append({
                'Brand': brand,
                'Year': year,
                'Paid Media (%)': (paid_media_spend_total / total_spend * 100) if total_spend else 0,
                'Promo (%)': (promo_spend_total / total_spend * 100) if total_spend else 0,
                'Retail (%)': (retail_spend_total / total_spend * 100) if total_spend else 0,
                'Total Spend': total_spend,
                'Total Sales': total_sales
            })
        
        return pd.DataFrame(results).set_index(['Brand', 'Year'])
    
    except Exception as e:
        logger.error(f"Error in calculate_media_spend_by_brand_year: {str(e)}")
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

def format_number(value: Any) -> str:
    """
    Convert numbers to M/B format and handle non-numerical values.
    
    Examples:
        1234567 -> "1.23M"
        1234567890 -> "1.23B"
        "Brand1" -> "Brand1"
        12.34 -> "12.34"
        None -> ""
    """
    try:
        # Handle None, empty strings, and non-numeric values
        if pd.isna(value) or value == "":
            return ""
        if not isinstance(value, (int, float)):
            return str(value)
        
        # Convert number to M/B format
        n = float(value)
        if abs(n) >= 1e9:
            return f"{n/1e9:.2f}B"
        elif abs(n) >= 1e6:
            return f"{n/1e6:.2f}M"
        elif abs(n) >= 1e3:
            return f"{n/1e3:.2f}K"
        else:
            return f"{n:,.2f}"
    except:
        return str(value)

def format_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format all numeric columns in the DataFrame.
    Preserves the original DataFrame and returns a formatted copy.
    """
    formatted_df = df.copy()
    
    # Format each column
    for col in formatted_df.columns:
        if col in ['hierarchy', 'year', 'Brand', 'Year']:  # Skip index columns
            continue
        formatted_df[col] = formatted_df[col].apply(format_number)
    
    return formatted_df

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
        
        # Get media categories
        col_categories = get_media_cols(df_new)

        # Calculate brand-wise media spend percentages
        old_brandwise = calculate_media_spend_by_brand(df_old)
        new_brandwise = calculate_media_spend_by_brand(df_new)
        
        # Create comparison DataFrame
        brand_components = pd.DataFrame({
            'Paid Media (%) Old': old_brandwise['Paid Media (%)'],
            'Paid Media (%) New': new_brandwise['Paid Media (%)'],
            'Paid Media Diff (%)': new_brandwise['Paid Media (%)'] - old_brandwise['Paid Media (%)'],
            'Promo (%) Old': old_brandwise['Promo (%)'],
            'Promo (%) New': new_brandwise['Promo (%)'],
            'Promo Diff (%)': new_brandwise['Promo (%)'] - old_brandwise['Promo (%)'],
            'Retail (%) Old': old_brandwise['Retail (%)'],
            'Retail (%) New': new_brandwise['Retail (%)'],
            'Retail Diff (%)': new_brandwise['Retail (%)'] - old_brandwise['Retail (%)'],
            'Total Spend Old': old_brandwise['Total Spend'],
            'Total Spend New': new_brandwise['Total Spend'],
            'Total Spend Diff': new_brandwise['Total Spend'] - old_brandwise['Total Spend'],
            'Total Sales Old': old_brandwise['Total Sales'],
            'Total Sales New': new_brandwise['Total Sales'],
            'Total Sales Diff': new_brandwise['Total Sales'] - old_brandwise['Total Sales']
        }).round(2).reset_index()

        # Calculate year-wise media spend percentages
        old_yearwise = calculate_media_spend_by_year(df_old)
        new_yearwise = calculate_media_spend_by_year(df_new)
        
        # Create year-wise comparison
        year_components = pd.DataFrame({
            'Paid Media (%) Old': old_yearwise['Paid Media (%)'],
            'Paid Media (%) New': new_yearwise['Paid Media (%)'],
            'Paid Media Diff (%)': new_yearwise['Paid Media (%)'] - old_yearwise['Paid Media (%)'],
            'Promo (%) Old': old_yearwise['Promo (%)'],
            'Promo (%) New': new_yearwise['Promo (%)'],
            'Promo Diff (%)': new_yearwise['Promo (%)'] - old_yearwise['Promo (%)'],
            'Retail (%) Old': old_yearwise['Retail (%)'],
            'Retail (%) New': new_yearwise['Retail (%)'],
            'Retail Diff (%)': new_yearwise['Retail (%)'] - old_yearwise['Retail (%)'],
            'Total Spend Old': old_yearwise['Total Spend'],
            'Total Spend New': new_yearwise['Total Spend'],
            'Total Spend Diff': new_yearwise['Total Spend'] - old_yearwise['Total Spend'],
            'Total Sales Old': old_yearwise['Total Sales'],
            'Total Sales New': new_yearwise['Total Sales'],
            'Total Sales Diff': new_yearwise['Total Sales'] - old_yearwise['Total Sales']
        }).round(2).reset_index()

        # Calculate brand-year media spend percentages
        old_brand_year = calculate_media_spend_by_brand_year(df_old)
        new_brand_year = calculate_media_spend_by_brand_year(df_new)
        
        # Create brand-year comparison
        brand_year_components = pd.DataFrame({
            'Paid Media (%) Old': old_brand_year['Paid Media (%)'],
            'Paid Media (%) New': new_brand_year['Paid Media (%)'],
            'Paid Media Diff (%)': new_brand_year['Paid Media (%)'] - old_brand_year['Paid Media (%)'],
            'Promo (%) Old': old_brand_year['Promo (%)'],
            'Promo (%) New': new_brand_year['Promo (%)'],
            'Promo Diff (%)': new_brand_year['Promo (%)'] - old_brand_year['Promo (%)'],
            'Retail (%) Old': old_brand_year['Retail (%)'],
            'Retail (%) New': new_brand_year['Retail (%)'],
            'Retail Diff (%)': new_brand_year['Retail (%)'] - old_brand_year['Retail (%)'],
            'Total Spend Old': old_brand_year['Total Spend'],
            'Total Spend New': new_brand_year['Total Spend'],
            'Total Spend Diff': new_brand_year['Total Spend'] - old_brand_year['Total Spend'],
            'Total Sales Old': old_brand_year['Total Sales'],
            'Total Sales New': new_brand_year['Total Sales'],
            'Total Sales Diff': new_brand_year['Total Sales'] - old_brand_year['Total Sales']
        }).round(2).reset_index()
        
        # Format all DataFrames
        brand_components = format_dataframe(brand_components)
        year_components = format_dataframe(year_components)
        brand_year_components = format_dataframe(brand_year_components)
        

        # Prepare template data
        template_data = {
            'new_file_name': os.path.basename(new_file_path),
            'old_file_name': os.path.basename(old_file_path),
            'start_date': start_date if start_date else df_new['date'].min().date(),
            'end_date': end_date if end_date else df_new['date'].max().date(),
            'country_name': country_name,
            'missing_in_new': missing_in_new,
            'missing_in_old': missing_in_old,
            'media_comparison': brand_components.to_html(index=False, classes="display", table_id="media-comparison"),
            'yearly_comparison': year_components.to_html(index=False, classes="display", table_id="yearly-comparison"),
            'yearly_brand_comparison': brand_year_components.to_html(index=False, classes="display", table_id="yearly-brand-comparison"),
            'summary_table': summary_table.to_html(index=False, classes="display", table_id="summary-table"),
            'brand_wise_charts': chart_paths,
            'paid_media_cols': col_categories.paid_media_spend_cols,
            'promo_cols': col_categories.promo_spend_cols,
            'retail_cols': col_categories.retail_spend_cols,
            'total_sales_cols': col_categories.total_sales_cols
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
