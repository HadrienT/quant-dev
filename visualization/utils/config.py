import os
from dotenv import load_dotenv
import QuantLib as ql

load_dotenv()
fred_api_key = os.getenv("FRED_APY_KEY")
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/app/service-account-key.json"


calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
