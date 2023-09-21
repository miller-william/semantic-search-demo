import random
from datetime import datetime, timedelta, date

########################### Miscellaneous ###########################

def random_date(start_date=datetime(2023, 5, 1), end_date=datetime.today()):
    return start_date + timedelta(
        seconds=random.randint(0, int((end_date - start_date).total_seconds())))