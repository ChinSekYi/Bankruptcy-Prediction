import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.utils import print_bankruptcy_outcome

application = Flask(__name__)

app = application

## Route for a home page


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = CustomData(
            net_profit_total_assets=request.form.get("net_profit_total_assets"),
            total_liabilities_total_assets=request.form.get(
                "total_liabilities_total_assets"
            ),
            working_capital_total_assets=request.form.get(
                "working_capital_total_assets"
            ),
            current_assets_short_term_liabilities=request.form.get(
                "current_assets_short_term_liabilities"
            ),
            cash_short_term_securities_receivables_short_term_liabilities_operating_expenses_depreciation_365=request.form.get(
                "cash_short_term_securities_receivables_short_term_liabilities_operating_expenses_depreciation_365"
            ),
            retained_earnings_total_assets=request.form.get(
                "retained_earnings_total_assets"
            ),
            EBIT_total_assets=request.form.get("EBIT_total_assets"),
            book_value_of_equity_total_liabilities=request.form.get(
                "book_value_of_equity_total_liabilities"
            ),
            sales_total_assets=request.form.get("sales_total_assets"),
            equity_total_assets=request.form.get("equity_total_assets"),
            gross_profit_extraordinary_items_financial_expenses_total_assets=request.form.get(
                "gross_profit_extraordinary_items_financial_expenses_total_assets"
            ),
            gross_profit_short_term_liabilities=request.form.get(
                "gross_profit_short_term_liabilities"
            ),
            gross_profit_depreciation_sales=request.form.get(
                "gross_profit_depreciation_sales"
            ),
            gross_profit_interest_total_assets=request.form.get(
                "gross_profit_interest_total_assets"
            ),
            total_liabilities_365_gross_profit_depreciation=request.form.get(
                "total_liabilities_365_gross_profit_depreciation"
            ),
            gross_profit_depreciation_total_liabilities=request.form.get(
                "gross_profit_depreciation_total_liabilities"
            ),
            total_assets_total_liabilities=request.form.get(
                "total_assets_total_liabilities"
            ),
            gross_profit_total_assets=request.form.get("gross_profit_total_assets"),
            gross_profit_sales=request.form.get("gross_profit_sales"),
            inventory_365_sales=request.form.get("inventory_365_sales"),
            sales_n_sales_n_1=request.form.get("sales_n_sales_n_1"),
            profit_on_operating_activities_total_assets=request.form.get(
                "profit_on_operating_activities_total_assets"
            ),
            net_profit_sales=request.form.get("net_profit_sales"),
            gross_profit_3_years_total_assets=request.form.get(
                "gross_profit_3_years_total_assets"
            ),
            equity_share_capital_total_assets=request.form.get(
                "equity_share_capital_total_assets"
            ),
            net_profit_depreciation_total_liabilities=request.form.get(
                "net_profit_depreciation_total_liabilities"
            ),
            profit_on_operating_activities_financial_expenses=request.form.get(
                "profit_on_operating_activities_financial_expenses"
            ),
            working_capital_fixed_assets=request.form.get(
                "working_capital_fixed_assets"
            ),
            logarithm_total_assets=request.form.get("logarithm_total_assets"),
            total_liabilities_cash_sales=request.form.get(
                "total_liabilities_cash_sales"
            ),
            gross_profit_interest_sales=request.form.get("gross_profit_interest_sales"),
            current_liabilities_365_cost_of_products_sold=request.form.get(
                "current_liabilities_365_cost_of_products_sold"
            ),
            operating_expenses_short_term_liabilities=request.form.get(
                "operating_expenses_short_term_liabilities"
            ),
            operating_expenses_total_liabilities=request.form.get(
                "operating_expenses_total_liabilities"
            ),
            profit_on_sales_total_assets=request.form.get(
                "profit_on_sales_total_assets"
            ),
            total_sales_total_assets=request.form.get("total_sales_total_assets"),
            current_assets_inventories_long_term_liabilities=request.form.get(
                "current_assets_inventories_long_term_liabilities"
            ),
            constant_capital_total_assets=request.form.get(
                "constant_capital_total_assets"
            ),
            profit_on_sales_sales=request.form.get("profit_on_sales_sales"),
            current_assets_inventory_receivables_short_term_liabilities=request.form.get(
                "current_assets_inventory_receivables_short_term_liabilities"
            ),
            total_liabilities_profit_on_operating_activities_depreciation_12_365=request.form.get(
                "total_liabilities_profit_on_operating_activities_depreciation_12_365"
            ),
            profit_on_operating_activities_sales=request.form.get(
                "profit_on_operating_activities_sales"
            ),
            rotation_receivables_inventory_turnover_days=request.form.get(
                "rotation_receivables_inventory_turnover_days"
            ),
            receivables_365_sales=request.form.get("receivables_365_sales"),
            net_profit_inventory=request.form.get("net_profit_inventory"),
            current_assets_inventory_short_term_liabilities=request.form.get(
                "current_assets_inventory_short_term_liabilities"
            ),
            inventory_365_cost_of_products_sold=request.form.get(
                "inventory_365_cost_of_products_sold"
            ),
            EBITDA_profit_on_operating_activities_depreciation_total_assets=request.form.get(
                "EBITDA_profit_on_operating_activities_depreciation_total_assets"
            ),
            EBITDA_profit_on_operating_activities_depreciation_sales=request.form.get(
                "EBITDA_profit_on_operating_activities_depreciation_sales"
            ),
            current_assets_total_liabilities=request.form.get(
                "current_assets_total_liabilities"
            ),
            short_term_liabilities_total_assets=request.form.get(
                "short_term_liabilities_total_assets"
            ),
            short_term_liabilities_365_cost_of_products_sold=request.form.get(
                "short_term_liabilities_365_cost_of_products_sold"
            ),
            equity_fixed_assets=request.form.get("equity_fixed_assets"),
            constant_capital_fixed_assets=request.form.get(
                "constant_capital_fixed_assets"
            ),
            working_capital=request.form.get("working_capital"),
            sales_cost_of_products_sold_sales=request.form.get(
                "sales_cost_of_products_sold_sales"
            ),
            current_assets_inventory_short_term_liabilities_sales_gross_profit_depreciation=request.form.get(
                "current_assets_inventory_short_term_liabilities_sales_gross_profit_depreciation"
            ),
            total_costs_total_sales=request.form.get("total_costs_total_sales"),
            long_term_liabilities_equity=request.form.get(
                "long_term_liabilities_equity"
            ),
            sales_inventory=request.form.get("sales_inventory"),
            sales_receivables=request.form.get("sales_receivables"),
            short_term_liabilities_365_sales=request.form.get(
                "short_term_liabilities_365_sales"
            ),
            sales_short_term_liabilities=request.form.get(
                "sales_short_term_liabilities"
            ),
            sales_fixed_assets=request.form.get("sales_fixed_assets"),
        )

        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        pred_result = predict_pipeline.predict(pred_df)
        bankruptcy_outcome = print_bankruptcy_outcome(pred_result)

        return render_template(
            "home.html",
            pred_result=pred_result[0],
            bankruptcy_outcome=bankruptcy_outcome,
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
