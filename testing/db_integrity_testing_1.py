from supabase import create_client
import pandas as pd
import os
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('db_integrity_tests_1.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

class DatabaseIntegrityTester:
    def __init__(self):
        self.supabase = create_client(url, key)
        self.test_results = []
    
    def get_table_data(self, table_name: str, select_fields: str = "*") -> pd.DataFrame:
        """Fetch all data from a table in batches"""
        batch_size = 1000
        offset = 0
        all_data = []

        while True:
            response = (
                self.supabase.table(table_name)
                .select(select_fields)
                .range(offset, offset + batch_size - 1)
                .execute()
            )
            data = response.data
            if not data:
                break
            all_data.extend(data)
            offset += batch_size

        return pd.DataFrame(all_data)
    
    def log_test_result(self, test_name: str, passed: bool, message: str, details: Any = None):
        """Log test results"""
        result = {
            'timestamp': datetime.now().isoformat(),
            'test_name': test_name,
            'passed': passed,
            'message': message,
            'details': details
        }
        self.test_results.append(result)
        
        status = "PASS" if passed else "FAIL"
        logger.info(f"[{status}] {test_name}: {message}")
        if details and not passed:
            logger.info(f"Details: {details}")
    
    def test_table_exists(self, table_name: str) -> bool:
        """Test if table exists and is accessible"""
        try:
            response = self.supabase.table(table_name).select("*").limit(1).execute()
            self.log_test_result(
                f"Table Existence - {table_name}",
                True,
                f"Table '{table_name}' exists and is accessible"
            )
            return True
        except Exception as e:
            self.log_test_result(
                f"Table Existence - {table_name}",
                False,
                f"Table '{table_name}' is not accessible",
                str(e)
            )
            return False
    
    def test_primary_key_constraints(self, table_name: str, pk_columns: List[str]) -> bool:
        """Test primary key uniqueness"""
        try:
            df = self.get_table_data(table_name, ",".join(pk_columns))
            
            if df.empty:
                self.log_test_result(
                    f"Primary Key Test - {table_name}",
                    True,
                    f"Table is empty, primary key constraint cannot be violated"
                )
                return True
            
            # Check for null values in primary key columns
            null_pk_count = df[pk_columns].isnull().any(axis=1).sum()
            if null_pk_count > 0:
                self.log_test_result(
                    f"Primary Key Null Test - {table_name}",
                    False,
                    f"Found {null_pk_count} rows with null values in primary key columns",
                    f"Primary key columns: {pk_columns}"
                )
                return False
            
            # Check for duplicate primary keys
            total_rows = len(df)
            unique_rows = df.drop_duplicates(subset=pk_columns).shape[0]
            
            if total_rows == unique_rows:
                self.log_test_result(
                    f"Primary Key Uniqueness - {table_name}",
                    True,
                    f"All {total_rows} rows have unique primary keys"
                )
                return True
            else:
                duplicates = total_rows - unique_rows
                self.log_test_result(
                    f"Primary Key Uniqueness - {table_name}",
                    False,
                    f"Found {duplicates} duplicate primary key combinations",
                    f"Total rows: {total_rows}, Unique combinations: {unique_rows}"
                )
                return False
                
        except Exception as e:
            self.log_test_result(
                f"Primary Key Test - {table_name}",
                False,
                f"Error testing primary key constraints",
                str(e)
            )
            return False
    
    def test_data_types_and_ranges(self, table_name: str, column_specs: Dict[str, Dict]) -> bool:
        """Test data types and value ranges"""
        try:
            df = self.get_table_data(table_name)
            
            if df.empty:
                self.log_test_result(
                    f"Data Type Test - {table_name}",
                    True,
                    "Table is empty, no data type violations possible"
                )
                return True
            
            all_passed = True
            
            for column, specs in column_specs.items():
                if column not in df.columns:
                    continue
                
                # Test for required columns (not null constraints)
                if specs.get('required', False):
                    null_count = df[column].isnull().sum()
                    if null_count > 0:
                        self.log_test_result(
                            f"Not Null Constraint - {table_name}.{column}",
                            False,
                            f"Found {null_count} null values in required column",
                            f"Column: {column}"
                        )
                        all_passed = False
                    else:
                        self.log_test_result(
                            f"Not Null Constraint - {table_name}.{column}",
                            True,
                            f"No null values found in required column"
                        )
                
                # Test numeric ranges
                if 'min_value' in specs or 'max_value' in specs:
                    non_null_data = df[column].dropna()
                    if len(non_null_data) > 0:
                        if 'min_value' in specs:
                            min_violations = (non_null_data < specs['min_value']).sum()
                            if min_violations > 0:
                                self.log_test_result(
                                    f"Min Value Test - {table_name}.{column}",
                                    False,
                                    f"Found {min_violations} values below minimum {specs['min_value']}",
                                    f"Actual min: {non_null_data.min()}"
                                )
                                all_passed = False
                        
                        if 'max_value' in specs:
                            max_violations = (non_null_data > specs['max_value']).sum()
                            if max_violations > 0:
                                self.log_test_result(
                                    f"Max Value Test - {table_name}.{column}",
                                    False,
                                    f"Found {max_violations} values above maximum {specs['max_value']}",
                                    f"Actual max: {non_null_data.max()}"
                                )
                                all_passed = False
                
                # Test allowed values
                if 'allowed_values' in specs:
                    non_null_data = df[column].dropna()
                    if len(non_null_data) > 0:
                        invalid_values = ~non_null_data.isin(specs['allowed_values'])
                        invalid_count = invalid_values.sum()
                        if invalid_count > 0:
                            unique_invalid = non_null_data[invalid_values].unique()
                            self.log_test_result(
                                f"Allowed Values Test - {table_name}.{column}",
                                False,
                                f"Found {invalid_count} values not in allowed list",
                                f"Invalid values: {list(unique_invalid)[:10]}"  # Show first 10
                            )
                            all_passed = False
                        else:
                            self.log_test_result(
                                f"Allowed Values Test - {table_name}.{column}",
                                True,
                                f"All values are in allowed list"
                            )
            
            return all_passed
            
        except Exception as e:
            self.log_test_result(
                f"Data Type Test - {table_name}",
                False,
                f"Error testing data types and ranges",
                str(e)
            )
            return False
    
    def test_data_consistency(self, table_name: str) -> bool:
        """Test business logic and data consistency"""
        try:
            df = self.get_table_data(table_name)
            
            if df.empty:
                self.log_test_result(
                    f"Data Consistency - {table_name}",
                    True,
                    "Table is empty, no consistency issues possible"
                )
                return True
            
            all_passed = True
            
            if table_name == "state_market":
                # Test date logic
                if 'year' in df.columns and 'month' in df.columns:
                    invalid_months = ((df['month'] < 1) | (df['month'] > 12)).sum()
                    if invalid_months > 0:
                        self.log_test_result(
                            f"Month Range - {table_name}",
                            False,
                            f"Found {invalid_months} records with invalid months",
                            f"Valid range: 1-12"
                        )
                        all_passed = False
                    
                    invalid_years = ((df['year'] < 1900) | (df['year'] > 2030)).sum()
                    if invalid_years > 0:
                        self.log_test_result(
                            f"Year Range - {table_name}",
                            False,
                            f"Found {invalid_years} records with unrealistic years",
                            f"Expected range: 1900-2030"
                        )
                        all_passed = False
                
                # Test price consistency
                price_columns = ['median_listing_price', 'average_listing_price']
                available_price_cols = [col for col in price_columns if col in df.columns]
                
                if len(available_price_cols) >= 2:
                    # Where both prices exist, average should generally be >= median
                    both_exist = df[available_price_cols].notna().all(axis=1)
                    if both_exist.any():
                        price_logic_violations = (
                            df.loc[both_exist, 'average_listing_price'] < 
                            df.loc[both_exist, 'median_listing_price'] * 0.5  # Allow some flexibility
                        ).sum()
                        
                        if price_logic_violations > 0:
                            self.log_test_result(
                                f"Price Logic - {table_name}",
                                False,
                                f"Found {price_logic_violations} records where average price is suspiciously lower than median",
                                "This might indicate data quality issues"
                            )
                            all_passed = False
            
            elif table_name == "predictions":
                # Test prediction date logic
                if 'year' in df.columns and 'month' in df.columns:
                    current_year = datetime.now().year
                    future_predictions = ((df['year'] > current_year + 5)).sum()
                    if future_predictions > 0:
                        self.log_test_result(
                            f"Prediction Range - {table_name}",
                            False,
                            f"Found {future_predictions} predictions more than 5 years in the future",
                            "This might be unrealistic for real estate predictions"
                        )
                        all_passed = False
                
                # Test market trend values
                if 'market_trend' in df.columns:
                    valid_trends = ['rising', 'declining', 'stable']
                    invalid_trends = ~df['market_trend'].isin(valid_trends + [None])
                    invalid_count = invalid_trends.sum()
                    if invalid_count > 0:
                        unique_invalid = df.loc[invalid_trends, 'market_trend'].unique()
                        self.log_test_result(
                            f"Market Trend Values - {table_name}",
                            False,
                            f"Found {invalid_count} records with invalid market trend values",
                            f"Invalid values: {list(unique_invalid)}"
                        )
                        all_passed = False
            
            if all_passed:
                self.log_test_result(
                    f"Data Consistency - {table_name}",
                    True,
                    "All data consistency checks passed"
                )
            
            return all_passed
            
        except Exception as e:
            self.log_test_result(
                f"Data Consistency - {table_name}",
                False,
                f"Error testing data consistency",
                str(e)
            )
            return False
    
    def generate_report(self) -> Dict:
        """Generate a summary report of all tests"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['passed'])
        failed_tests = total_tests - passed_tests
        
        report = {
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0
            },
            'failed_tests': [result for result in self.test_results if not result['passed']],
            'all_results': self.test_results
        }
        
        logger.info(f"\n{'='*50}")
        logger.info("DATABASE INTEGRITY TEST SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Success Rate: {report['summary']['success_rate']:.1f}%")
        
        if failed_tests > 0:
            logger.info(f"\nFAILED TESTS:")
            for result in report['failed_tests']:
                logger.info(f"- {result['test_name']}: {result['message']}")
        
        return report

def run_comprehensive_integrity_tests():
    """Run all database integrity tests"""
    tester = DatabaseIntegrityTester()
    
    # Define column specifications for validation
    state_market_specs = {
        'year': {'required': True, 'min_value': 1900, 'max_value': 2030},
        'month': {'required': True, 'min_value': 1, 'max_value': 12},
        'state': {'required': True},
        'median_listing_price': {'min_value': 0},
        'active_listing_count': {'min_value': 0},
        'median_days_on_market': {'min_value': 0},
        'new_listing_count': {'min_value': 0},
        'price_increased_count': {'min_value': 0},
        'price_reduced_count': {'min_value': 0},
        'pending_listing_count': {'min_value': 0},
        'median_listing_price_per_square_foot': {'min_value': 0},
        'median_square_feet': {'min_value': 0},
        'average_listing_price': {'min_value': 0},
        'total_listing_count': {'min_value': 0}
    }
    
    predictions_specs = {
        'year': {'required': True, 'min_value': 2020, 'max_value': 2035},
        'month': {'required': True, 'min_value': 1, 'max_value': 12},
        'state': {'required': True},
        'median_listing_price': {'min_value': 0},
        'average_listing_price': {'min_value': 0},
        'median_listing_price_per_square_foot': {'min_value': 0},
        'total_listing_count': {'min_value': 0},
        'median_days_on_market': {'min_value': 0},
        'market_trend': {'allowed_values': ['rising', 'declining', 'stable']}
    }
    
    logger.info("Starting comprehensive database integrity tests...")
    
    # Test 1: Table existence
    logger.info("\n1. Testing table existence...")
    tester.test_table_exists("state_market")
    tester.test_table_exists("predictions")
    
    # Test 2: Primary key constraints
    logger.info("\n2. Testing primary key constraints...")
    tester.test_primary_key_constraints("state_market", ["year", "month", "state"])
    tester.test_primary_key_constraints("predictions", ["year", "month", "state"])
    
    # Test 3: Data types and ranges
    logger.info("\n3. Testing data types and ranges...")
    tester.test_data_types_and_ranges("state_market", state_market_specs)
    tester.test_data_types_and_ranges("predictions", predictions_specs)

    
    # Test 4: Data consistency and business logic
    logger.info("\n4. Testing data consistency...")
    tester.test_data_consistency("state_market")
    tester.test_data_consistency("predictions")
    
    # Generate final report
    logger.info("\n6. Generating report...")
    report = tester.generate_report()
    
    return report

# Legacy function for backward compatibility
def get_supabase_data():
    """Legacy function - use DatabaseIntegrityTester.get_table_data() instead"""
    tester = DatabaseIntegrityTester()
    return tester.get_table_data("state_market")

if __name__ == "__main__":
    report = run_comprehensive_integrity_tests()
    
    # Save detailed report to file
    import json
    with open('integrity_test_report_1.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nDetailed report saved to 'integrity_test_report_1.json'")
    print(f"Test logs saved to 'db_integrity_tests_1.log'")