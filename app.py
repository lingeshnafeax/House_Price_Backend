from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

# Load the trained model and scaler
final_model = joblib.load("final_model.pkl")
scaler = joblib.load("scaler.pkl")


@app.route("/", methods=["POST"])
def predict():
    data = request.get_json()
    # Get data from the request
    prediction = pd.DataFrame(
        columns=[
            "INT_SQFT",
            "DIST_MAINROAD",
            "N_BEDROOM",
            "N_BATHROOM",
            "PARK_FACIL",
            "QS_ROOMS",
            "QS_BATHROOM",
            "QS_BEDROOM",
            "QS_OVERALL",
            "REG_FEE",
            "COMMIS",
            "AREA_Adyar",
            "AREA_Anna Nagar",
            "AREA_Chrompet",
            "AREA_KK Nagar",
            "AREA_Karapakam",
            "AREA_T Nagar",
            "AREA_Velachery",
            "SALE_COND_AbNormal",
            "SALE_COND_AdjLand",
            "SALE_COND_Family",
            "SALE_COND_Normal Sale",
            "SALE_COND_Partial",
            "BUILDTYPE_Commercial",
            "BUILDTYPE_House",
            "BUILDTYPE_Others",
            "UTILITY_AVAIL_All Pub",
            "UTILITY_AVAIL_ELO",
            "UTILITY_AVAIL_NoSeWa",
            "STREET_Gravel",
            "STREET_No Access",
            "STREET_Paved",
            "MZZONE_A",
            "MZZONE_C",
            "MZZONE_I",
            "MZZONE_RH",
            "MZZONE_RL",
            "MZZONE_RM",
        ]
    )
    temp_pred = pd.DataFrame(data)
    temp_pred = pd.get_dummies(
        data=temp_pred,
        columns=["AREA", "SALE_COND", "BUILDTYPE", "UTILITY_AVAIL", "STREET", "MZZONE"],
        dtype=int,
    )
    final_pred = pd.concat([prediction, temp_pred])
    final_pred.fillna(inplace=True, value=0)
    final_pred = final_pred.drop(
        [
            "AREA_Adyar",
            "AREA_Anna Nagar",
            "AREA_T Nagar",
            "MZZONE_A",
            "MZZONE_C",
            "MZZONE_I",
        ],
        axis=1,
    )
    final_pred = pd.DataFrame(scaler.transform(final_pred), columns=final_pred.columns)
    pred = final_model.predict(final_pred)

    # Convert numpy array to a serializable format (nested list)
    pred_serializable = pred.tolist()

    # Return the prediction
    return jsonify({"prediction": pred_serializable[0]})


if __name__ == "__main__":
    app.run(debug=True)
