import sys

import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass


class CustomData:
    """
    responsible for taking the inputs that we are giving in the HTML to the backend
    """

    def __init__(
        self,
        net_profit_total_assets: float,
        total_liabilities_total_assets: float,
        working_capital_total_assets: float,
        current_assets_short_term_liabilities: float,
        cash_short_term_securities_receivables_short_term_liabilities_operating_expenses_depreciation_365: float,
        retained_earnings_total_assets: float,
        EBIT_total_assets: float,
        book_value_of_equity_total_liabilities: float,
        sales_total_assets: float,
        equity_total_assets: float,
        gross_profit_extraordinary_items_financial_expenses_total_assets: float,
        gross_profit_short_term_liabilities: float,
        gross_profit_depreciation_sales: float,
        gross_profit_interest_total_assets: float,
        total_liabilities_365_gross_profit_depreciation: float,
        gross_profit_depreciation_total_liabilities: float,
        total_assets_total_liabilities: float,
        gross_profit_total_assets: float,
        gross_profit_sales: float,
        inventory_365_sales: float,
        sales_n_sales_n_1: float,
        profit_on_operating_activities_total_assets: float,
        net_profit_sales: float,
        gross_profit_3_years_total_assets: float,
        equity_share_capital_total_assets: float,
        net_profit_depreciation_total_liabilities: float,
        profit_on_operating_activities_financial_expenses: float,
        working_capital_fixed_assets: float,
        logarithm_total_assets: float,
        total_liabilities_cash_sales: float,
        gross_profit_interest_sales: float,
        current_liabilities_365_cost_of_products_sold: float,
        operating_expenses_short_term_liabilities: float,
        operating_expenses_total_liabilities: float,
        profit_on_sales_total_assets: float,
        total_sales_total_assets: float,
        current_assets_inventories_long_term_liabilities: float,
        constant_capital_total_assets: float,
        profit_on_sales_sales: float,
        current_assets_inventory_receivables_short_term_liabilities: float,
        total_liabilities_profit_on_operating_activities_depreciation_12_365: float,
        profit_on_operating_activities_sales: float,
        rotation_receivables_inventory_turnover_days: float,
        receivables_365_sales: float,
        net_profit_inventory: float,
        current_assets_inventory_short_term_liabilities: float,
        inventory_365_cost_of_products_sold: float,
        EBITDA_profit_on_operating_activities_depreciation_total_assets: float,
        EBITDA_profit_on_operating_activities_depreciation_sales: float,
        current_assets_total_liabilities: float,
        short_term_liabilities_total_assets: float,
        short_term_liabilities_365_cost_of_products_sold: float,
        equity_fixed_assets: float,
        constant_capital_fixed_assets: float,
        working_capital: float,
        sales_cost_of_products_sold_sales: float,
        current_assets_inventory_short_term_liabilities_sales_gross_profit_depreciation: float,
        total_costs_total_sales: float,
        long_term_liabilities_equity: float,
        sales_inventory: float,
        sales_receivables: float,
        short_term_liabilities_365_sales: float,
        sales_short_term_liabilities: float,
        sales_fixed_assets: float,
        class_value: float,
    ):

        self.net_profit_total_assets = net_profit_total_assets
        self.total_liabilities_total_assets = total_liabilities_total_assets
        self.working_capital_total_assets = working_capital_total_assets
        self.current_assets_short_term_liabilities = (
            current_assets_short_term_liabilities
        )
        self.cash_short_term_securities_receivables_short_term_liabilities_operating_expenses_depreciation_365 = cash_short_term_securities_receivables_short_term_liabilities_operating_expenses_depreciation_365
        self.retained_earnings_total_assets = retained_earnings_total_assets
        self.EBIT_total_assets = EBIT_total_assets
        self.book_value_of_equity_total_liabilities = (
            book_value_of_equity_total_liabilities
        )
        self.sales_total_assets = sales_total_assets
        self.equity_total_assets = equity_total_assets
        self.gross_profit_extraordinary_items_financial_expenses_total_assets = (
            gross_profit_extraordinary_items_financial_expenses_total_assets
        )
        self.gross_profit_short_term_liabilities = gross_profit_short_term_liabilities
        self.gross_profit_depreciation_sales = gross_profit_depreciation_sales
        self.gross_profit_interest_total_assets = gross_profit_interest_total_assets
        self.total_liabilities_365_gross_profit_depreciation = (
            total_liabilities_365_gross_profit_depreciation
        )
        self.gross_profit_depreciation_total_liabilities = (
            gross_profit_depreciation_total_liabilities
        )
        self.total_assets_total_liabilities = total_assets_total_liabilities
        self.gross_profit_total_assets = gross_profit_total_assets
        self.gross_profit_sales = gross_profit_sales
        self.inventory_365_sales = inventory_365_sales
        self.sales_n_sales_n_1 = sales_n_sales_n_1
        self.profit_on_operating_activities_total_assets = (
            profit_on_operating_activities_total_assets
        )
        self.net_profit_sales = net_profit_sales
        self.gross_profit_3_years_total_assets = gross_profit_3_years_total_assets
        self.equity_share_capital_total_assets = equity_share_capital_total_assets
        self.net_profit_depreciation_total_liabilities = (
            net_profit_depreciation_total_liabilities
        )
        self.profit_on_operating_activities_financial_expenses = (
            profit_on_operating_activities_financial_expenses
        )
        self.working_capital_fixed_assets = working_capital_fixed_assets
        self.logarithm_total_assets = logarithm_total_assets
        self.total_liabilities_cash_sales = total_liabilities_cash_sales
        self.gross_profit_interest_sales = gross_profit_interest_sales
        self.current_liabilities_365_cost_of_products_sold = (
            current_liabilities_365_cost_of_products_sold
        )
        self.operating_expenses_short_term_liabilities = (
            operating_expenses_short_term_liabilities
        )
        self.operating_expenses_total_liabilities = operating_expenses_total_liabilities
        self.profit_on_sales_total_assets = profit_on_sales_total_assets
        self.total_sales_total_assets = total_sales_total_assets
        self.current_assets_inventories_long_term_liabilities = (
            current_assets_inventories_long_term_liabilities
        )
        self.constant_capital_total_assets = constant_capital_total_assets
        self.profit_on_sales_sales = profit_on_sales_sales
        self.current_assets_inventory_receivables_short_term_liabilities = (
            current_assets_inventory_receivables_short_term_liabilities
        )
        self.total_liabilities_profit_on_operating_activities_depreciation_12_365 = (
            total_liabilities_profit_on_operating_activities_depreciation_12_365
        )
        self.profit_on_operating_activities_sales = profit_on_operating_activities_sales
        self.rotation_receivables_inventory_turnover_days = (
            rotation_receivables_inventory_turnover_days
        )
        self.receivables_365_sales = receivables_365_sales
        self.net_profit_inventory = net_profit_inventory
        self.current_assets_inventory_short_term_liabilities = (
            current_assets_inventory_short_term_liabilities
        )
        self.inventory_365_cost_of_products_sold = inventory_365_cost_of_products_sold
        self.EBITDA_profit_on_operating_activities_depreciation_total_assets = (
            EBITDA_profit_on_operating_activities_depreciation_total_assets
        )
        self.EBITDA_profit_on_operating_activities_depreciation_sales = (
            EBITDA_profit_on_operating_activities_depreciation_sales
        )
        self.current_assets_total_liabilities = current_assets_total_liabilities
        self.short_term_liabilities_total_assets = short_term_liabilities_total_assets
        self.short_term_liabilities_365_cost_of_products_sold = (
            short_term_liabilities_365_cost_of_products_sold
        )
        self.equity_fixed_assets = equity_fixed_assets
        self.constant_capital_fixed_assets = constant_capital_fixed_assets
        self.working_capital = working_capital
        self.sales_cost_of_products_sold_sales = sales_cost_of_products_sold_sales
        self.current_assets_inventory_short_term_liabilities_sales_gross_profit_depreciation = current_assets_inventory_short_term_liabilities_sales_gross_profit_depreciation
        self.total_costs_total_sales = total_costs_total_sales
        self.long_term_liabilities_equity = long_term_liabilities_equity
        self.sales_inventory = sales_inventory
        self.sales_receivables = sales_receivables
        self.short_term_liabilities_365_sales = short_term_liabilities_365_sales
        self.sales_short_term_liabilities = sales_short_term_liabilities
        self.sales_fixed_assets = sales_fixed_assets
        self.class_value = class_value
