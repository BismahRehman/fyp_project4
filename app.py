import streamlit as st
import pickle
import pandas as pd
import joblib

# Load the model and preprocessing objects
model = pickle.load(open('model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
expected_columns = pickle.load(open('model_columns.pkl', 'rb'))  # list of one-hot encoded columns

# Streamlit UI
st.title("💳 Credit Card Fraud Detection")

# Input fields
gender = st.selectbox("Gender", ['F', 'M'])
channel = st.selectbox("Channel", ['Online', 'In-store'])
type_of_card = st.selectbox("Type of Card", ['Visa', 'MasterCard'])
device = st.selectbox("Device Used", ['Mobile', 'Desktop'])
category = st.selectbox("Category", ['Shopping', 'Food', 'Travel', 'Other'])
job = st.selectbox("Job Title", ['Film/video editor', 'Exhibition designer', 'Surveyor'])
city = st.selectbox("City", [
    'city_Achille', 'city_Acworth', 'city_Adams', 'city_Afton', 'city_Akron', 'city_Albany','city_Albuquerque','city_Alder', 'city_Alexandria',
    'city_Allenhurst', 'city_Allentown', 'city_Alpharetta', 'city_Altair', 'city_Altona', 'city_Altonah',
    'city_Alva', 'city_American Fork', 'city_Amorita', 'city_Andrews', 'city_Annapolis', 'city_Arcadia',
    'city_Arlington', 'city_Armagh', 'city_Armonk', 'city_Arnold', 'city_Arvada', 'city_Ash Flat', 'city_Ashfield',
    'city_Ashford', 'city_Atglen', 'city_Athena', 'city_Atlantic', 'city_Auburn', 'city_Aurora', 'city_Avera', 'city_Avoca',
    'city_Azusa', 'city_Bagley', 'city_Bailey', 'city_Barnard', 'city_Barneveld', 'city_Barnstable', 'city_Baroda',
    'city_Basye', 'city_Baton Rouge', 'city_Battle Creek', 'city_Bauxite', 'city_Bay Minette',
    'city_Beaver Falls', 'city_Beaverdam', 'city_Belfast', 'city_Belgrade', 'city_Bellmore', 'city_Belmond',
    'city_Belmont', 'city_Benton', 'city_Bessemer', 'city_Bethel', 'city_Bethel Springs', 'city_Big Creek',
    'city_Big Indian', 'city_Birmingham', 'city_Blackville', 'city_Blairstown', 'city_Blooming Grove',
    'city_Bolivar', 'city_Bolton', 'city_Bonfield', 'city_Bonita Springs', 'city_Boulder', 'city_Bowdoin', 'city_Boyd',
    'city_Bradley', 'city_Brainard', 'city_Brandon', 'city_Brantley', 'city_Breesport', 'city_Bridgeport',
    'city_Bridger', 'city_Brinson', 'city_Bristol', 'city_Bristow', 'city_Bronx', 'city_Brooklyn', 'city_Broomfield',
    'city_Browning', 'city_Brownville', 'city_Brunson', 'city_Bryant', 'city_Burbank', 'city_Burke', 'city_Burlington',
    'city_Burns Flat', 'city_Burrton', 'city_Bynum', 'city_Cadiz', 'city_Camden', 'city_Campbell', 'city_Canton',
    'city_Cape Coral', 'city_Cardwell', 'city_Carlisle', 'city_Carlotta', 'city_Carroll', 'city_Cascade Locks',
    'city_Cass', 'city_Catawba', 'city_Cazenovia', 'city_Cecilton', 'city_Cedar', 'city_Center Point',
    'city_Center Tuftonboro', 'city_Centerview', 'city_Central', 'city_Chatham', 'city_Cherokee Village',
    'city_Chester Heights', 'city_Christine', 'city_Churubusco', 'city_Cisco', 'city_Claremont', 'city_Clarinda',
    'city_Clarks Mills', 'city_Clarksville', 'city_Clay Center', 'city_Clayton', 'city_Clearwater',
    'city_Cleveland', 'city_Clutier', 'city_Cochranton', 'city_Coffeeville', 'city_Cokeburg', 'city_Coleharbor',
    'city_Coleman', 'city_Collettsville', 'city_Colorado Springs', 'city_Columbia', 'city_Comfort',
    'city_Comfrey', 'city_Conway', 'city_Cord', 'city_Corona', 'city_Corriganville', 'city_Corsica', 'city_Cottekill',
    'city_Cowlesville', 'city_Coyle', 'city_Craig', 'city_Creedmoor', 'city_Creola', 'city_Cressona', 'city_Cromona',
    'city_Cross Plains', 'city_Crownpoint', 'city_Curlew', 'city_Cuthbert', 'city_Cuyahoga Falls', 'city_Dallas',
    'city_Daly City', 'city_Daniels', 'city_Darien', 'city_De Lancey', 'city_De Queen', 'city_De Soto', 'city_De Witt',
    'city_Deadwood', 'city_Deane', 'city_Delhi', 'city_Dell City', 'city_Deltona', 'city_Denham Springs',
    'city_Des Moines', 'city_Desdemona', 'city_Detroit', 'city_Diamond', 'city_Dieterich', 'city_Doe Hill',
    'city_Dongola', 'city_Drakes Branch', 'city_Dresden', 'city_Dubre', 'city_Dumont', 'city_Duncan', 'city_Dunlevy',
    'city_Eagarville', 'city_East Andover', 'city_East Canaan', 'city_East Rochester', 'city_Easton',
    'city_Edinburg', 'city_Edisto Island', 'city_Edmond', 'city_Egan', 'city_El Paso', 'city_Elberta', 'city_Eldridge',
    'city_Elizabeth', 'city_Elizabethtown', 'city_Elk Rapids', 'city_Elkhart', 'city_Enola', 'city_Esbon',
    'city_Espanola', 'city_Etlan', 'city_Eugene', 'city_Eureka', 'city_Fairhope', 'city_Fairview', 'city_Falconer',
    'city_Falls Church', 'city_Falls City', 'city_Falmouth', 'city_Farmington', 'city_Fayetteville',
    'city_Fenelton', 'city_Fiddletown', 'city_Fields Landing', 'city_Florence', 'city_Ford', 'city_Fordoche',
    'city_Fort Myers', 'city_Fort Washakie', 'city_Freedom', 'city_Fullerton', 'city_Fulton', 'city_Gadsden',
    'city_Gardiner', 'city_Garfield', 'city_Garrattsville', 'city_Georgetown', 'city_Gibsonville', 'city_Girard',
    'city_Glade Spring', 'city_Glen Rock', 'city_Glendale', 'city_Goodrich', 'city_Goreville', 'city_Grand Bay',
    'city_Grand Junction', 'city_Grand Ridge', 'city_Grandview', 'city_Graniteville', 'city_Grant',
    'city_Grantham', 'city_Grassflat', 'city_Great Mills', 'city_Greenbush', 'city_Greendale', 'city_Greenview',
    'city_Greenville', 'city_Greenwich', 'city_Greenwood', 'city_Gregory', 'city_Grenada', 'city_Gretna',
    'city_Grifton', 'city_Grover', 'city_Guthrie', 'city_Hahira', 'city_Halma', 'city_Halstad', 'city_Hampton',
    'city_Hancock', 'city_Hannawa Falls', 'city_Harborcreek', 'city_Harper', 'city_Harrington Park',
    'city_Harrodsburg', 'city_Hartford', 'city_Harwood', 'city_Hatch', 'city_Haw River', 'city_Hawley', 'city_Hazel',
    'city_Heart Butte', 'city_Hedley', 'city_Hedrick', 'city_Heidelberg', 'city_Heiskell', 'city_Heislerville',
    'city_Helm', 'city_Henderson', 'city_Hewitt', 'city_Higganum', 'city_High Rolls Mountain Park',
    'city_Highland', 'city_Hills', 'city_Hinckley', 'city_Hinesburg', 'city_Holcomb', 'city_Holliday', 'city_Holloway',
    'city_Holstein', 'city_Honokaa', 'city_Hooper', 'city_Hopkins', 'city_Houston', 'city_Hovland', 'city_Howells',
    'city_Howes Cave', 'city_Hudson', 'city_Humble', 'city_Humboldt', 'city_Huntington Beach', 'city_Huntsville',
    'city_Hurley', 'city_Hurricane', 'city_Iliff', 'city_Independence', 'city_Indian Wells', 'city_Indianapolis',
    'city_Irwinton', 'city_Issaquah', 'city_Jackson', 'city_Jaffrey', 'city_Jay', 'city_Jefferson', 'city_Jelm',
    'city_Jermyn', 'city_Johns Island', 'city_Joliet', 'city_Jones', 'city_Jordan Valley', 'city_Jordanville',
    'city_Juliette', 'city_June Lake', 'city_Kansas City', 'city_Karnack', 'city_Keisterville', 'city_Keller',
    'city_Kenner', 'city_Kensington', 'city_Kent', 'city_Key West', 'city_Kilgore', 'city_Kings Bay',
    'city_Kingsford Heights', 'city_Kingsport', 'city_Kingsville', 'city_Kirby', 'city_Kirk', 'city_Kirkwood',
    'city_Kirtland', 'city_Kissee Mills', 'city_Kittery Point', 'city_Knoxville', 'city_Lagrange',
    'city_Laguna Hills', 'city_Lahoma', 'city_Lake Jackson', 'city_Lake Oswego', 'city_Lakeland',
    'city_Lakeport', 'city_Lakeview', 'city_Lamberton', 'city_Lanark Village', 'city_Lane', 'city_Laramie',
    'city_Laredo', 'city_Lawn', 'city_Lawrence', 'city_Lebanon', 'city_Leetsdale', 'city_Leo', 'city_Leonard',
    'city_Lexington', 'city_Lima', 'city_Linthicum Heights', 'city_Lithopolis', 'city_Littleton', 'city_Livonia',
    'city_Llano', 'city_Loami', 'city_Logan', 'city_Lohrville', 'city_Lolita', 'city_Lomax', 'city_Lonetree',
    'city_Lonsdale', 'city_Lorenzo', 'city_Los Angeles', 'city_Louisiana', 'city_Louisville', 'city_Loving',
    'city_Lowell', 'city_Lowville', 'city_Loxahatchee', 'city_Lubbock', 'city_Luray', 'city_Luxembourg',
    'city_Lynchburg', 'city_Lyndon', 'city_Macon', 'city_Madison', 'city_Magalia', 'city_Mahwah', 'city_Makawao',
    'city_Malden', 'city_Malibu', 'city_Malvern', 'city_Marathon', 'city_Maricopa', 'city_Marion', 'city_Marshfield',
    'city_Martinsville', 'city_Marysville', 'city_Mason', 'city_Massillon', 'city_Matawan', 'city_Mayville',
    'city_McAllen', 'city_McMinnville', 'city_Meadville', 'city_Mechanicsville', 'city_Medford', 'city_Memphis',
    'city_Mendota', 'city_Mentone', 'city_Meridian', 'city_Merrill', 'city_Mesa', 'city_Mesquite', 'city_Metropolis',
    'city_Midland', 'city_Middletown', 'city_Milford', 'city_Millbrook', 'city_Millburn', 'city_Millville',
    'city_Milpitas', 'city_Minneapolis', 'city_Minot', 'city_Missoula', 'city_Mobile', 'city_Modesto', 'city_Moline',
    'city_Monroe', 'city_Montclair', 'city_Montgomery', 'city_Montrose', 'city_Moore', 'city_Moorhead', 'city_Morrisville',
    'city_Morton', 'city_Moscow', 'city_Mount Pleasant', 'city_Mount Vernon', 'city_Muncie', 'city_Murfreesboro',
    'city_Muscatine', 'city_Muskegon', 'city_Myrtle Beach', 'city_Nacogdoches', 'city_Naperville', 'city_Naples',
    'city_Nashville', 'city_Natchez', 'city_New Braunfels', 'city_New Haven', 'city_New Orleans', 'city_New York',
    'city_Newark', 'city_Newport', 'city_Newton', 'city_Nicholasville', 'city_Niles', 'city_Norfolk', 'city_Norman',
    'city_Northampton', 'city_Northfield', 'city_Northport', 'city_Northville', 'city_Norwood', 'city_Oakland',
    'city_Oakley', 'city_Ocean City', 'city_Oceanside', 'city_Ogden', 'city_Oklahoma City', 'city_Olympia',
    'city_Omaha', 'city_Orange', 'city_Orlando', 'city_Oro Valley', 'city_Oshkosh', 'city_Otay Mesa', 'city_Oxnard',
    'city_Pacifica', 'city_Palatine', 'city_Palisades Park', 'city_Palm Bay', 'city_Palm Coast', 'city_Palm Desert',
    'city_Palm Springs', 'city_Panama City', 'city_Parsons', 'city_Pasco', 'city_Pasadena', 'city_Pawtucket',
    'city_Peabody', 'city_Pekin', 'city_Pelham', 'city_Pembroke Pines', 'city_Penn Valley', 'city_Pensacola',
    'city_Peoria', 'city_Phoenix', 'city_Pico Rivera', 'city_Pine Bluff', 'city_Pinehurst', 'city_Pittsburgh',
    'city_Plano', 'city_Plattsburgh', 'city_Plymouth', 'city_Pocomoke City', 'city_Pocatello', 'city_Pomona',
    'city_Pontiac', 'city_Port Arthur', 'city_Port Orange', 'city_Port St. Lucie', 'city_Portage', 'city_Portland',
    'city_Preston', 'city_Princeton', 'city_Providence', 'city_Provo', 'city_Pueblo', 'city_Puyallup', 'city_Quincy',
    'city_Racine', 'city_Raleigh', 'city_Rapid City', 'city_Red Bluff', 'city_Redondo Beach', 'city_Redwood City',
    'city_Refugio', 'city_Reno', 'city_Reseda', 'city_Reseda', 'city_Research Triangle Park', 'city_Reston',
    'city_Richmond', 'city_Richmond Hill', 'city_Richland', 'city_Ridgefield', 'city_Riverdale', 'city_Riverside',
    'city_Roanoke', 'city_Rock Hill', 'city_Rock Island', 'city_Rockford', 'city_Rockport', 'city_Rocky Mount',
    'city_Rolla', 'city_Rome', 'city_Roseburg', 'city_Roselle', 'city_Roswell', 'city_Russellville', 'city_Rutland',
    'city_Sacramento', 'city_Saginaw', 'city_Salem', 'city_Salina', 'city_Salisbury', 'city_San Antonio',
    'city_San Bernardino', 'city_San Diego', 'city_San Francisco', 'city_San Jose', 'city_San Luis Obispo',
    'city_San Marcos', 'city_San Mateo', 'city_San Rafael', 'city_San Ramon', 'city_Santa Ana', 'city_Santa Barbara',
    'city_Santa Clara', 'city_Santa Cruz', 'city_Santa Fe', 'city_Santa Monica', 'city_Santa Rosa', 'city_Sarasota',
    'city_Savannah', 'city_Schenectady', 'city_Schenley', 'city_Schroon Lake', 'city_Scottsdale', 'city_Seaside',
    'city_Sebring', 'city_Sedalia', 'city_Seneca', 'city_Seymour', 'city_Sherman', 'city_Shoreline', 'city_Short Hills',
    'city_Sioux City', 'city_Sioux Falls', 'city_Skidmore', 'city_Skokie', 'city_Slake', 'city_Smyrna', 'city_Snoqualmie',
    'city_Snohomish', 'city_Soil', 'city_Somerville', 'city_South Bend', 'city_South Boston', 'city_South Gate',
    'city_South Haven', 'city_South Lake Tahoe', 'city_South Lyon', 'city_South San Francisco', 'city_Southfield',
    'city_Springfield', 'city_St. Charles', 'city_St. Cloud', 'city_St. George', 'city_St. Louis', 'city_St. Paul',
    'city_St. Petersburg', 'city_Stafford', 'city_Stanford', 'city_Stanley', 'city_Staples', 'city_Stapleton',
    'city_Starke', 'city_Starr', 'city_Staten Island', 'city_Stateline', 'city_Stoughton', 'city_Strawberry',
    'city_Stroud', 'city_Strum', 'city_Sullivan', 'city_Summerville', 'city_Sun City', 'city_Sun Prairie', 'city_Sunnyvale',
    'city_Superior', 'city_Surprise', 'city_Sussex', 'city_Swainsboro', 'city_Syracuse', 'city_Tacoma',
    'city_Tallahassee', 'city_Tampa', 'city_Taylorsville', 'city_Tecumseh', 'city_Temecula', 'city_Tempe', 'city_Tennessee',
    'city_Terre Haute', 'city_The Dalles', 'city_Thornton', 'city_Thomasville', 'city_Thompson', 'city_Thornton',
    'city_Tiffin', 'city_Titusville', 'city_Toledo', 'city_Tomah', 'city_Tompkinsville', 'city_Tonka Bay',
    'city_Torrance', 'city_Town and Country', 'city_Townsend', 'city_Tracy', 'city_Trenton', 'city_Triadelphia',
    'city_Tucson', 'city_Tulsa', 'city_Tupelo', 'city_Turkey', 'city_Turlock', 'city_Tustin', 'city_Tuttle',
    'city_Twin Falls', 'city_Tyler', 'city_Utica', 'city_Uvalde', 'city_Vail', 'city_Van Nuys', 'city_Vancouver',
    'city_Vernon', 'city_Vero Beach', 'city_Victoria', 'city_Vicksburg', 'city_Vienna', 'city_Vincennes',
    'city_Virginia Beach', 'city_Vista', 'city_Waco', 'city_Wadsworth', 'city_Wake Forest', 'city_Walden',
    'city_Wallingford', 'city_Walnut Creek', 'city_Warren', 'city_Warrensburg', 'city_Warrenton', 'city_Warren',
    'city_Washington', 'city_Waterloo', 'city_Watertown', 'city_Waterville', 'city_Watkins Glen', 'city_Watkinsville',
    'city_Watsonville', 'city_Watsonville', 'city_Watsonville', 'city_Waverly', 'city_Waynesboro', 'city_Weatherford',
    'city_Webster', 'city_Wedowee', 'city_Weirton', 'city_Wellsville', 'city_Wells', 'city_Wellsburg', 'city_Wellington',
    'city_Wellston', 'city_Wellsboro', 'city_Wenatchee', 'city_West Allis', 'city_West Bend', 'city_Westbrook',
    'city_Westbury', 'city_Westminster', 'city_Westport', 'city_Westwood', 'city_Wheeling', 'city_White Plains',
    'city_White Sulphur Springs', 'city_Whitehall', 'city_Whitesboro', 'city_Whitestown', 'city_Wichita',
    'city_Wichita Falls', 'city_Williamsburg', 'city_Williamsport', 'city_Williston', 'city_Wilmington',
    'city_Wilson', 'city_Winchester', 'city_Windham', 'city_Windsor', 'city_Winnemucca', 'city_Winthrop',
    'city_Winter Haven', 'city_Winter Park', 'city_Winter Springs', 'city_Winston-Salem', 'city_Winterville',
    'city_Wisconsin Rapids', 'city_Woodbridge', 'city_Woodland', 'city_Woolwich', 'city_Worcester', 'city_Wyoming',
    'city_Yakima', 'city_York', 'city_Yorkville', 'city_Yuma'
])


state = st.selectbox("State", ['TX', 'NY', 'PA', 'CA', 'OH', 'MI', 'IL'])
amt = st.number_input("Amount", min_value=0.0)
login_attempts = st.slider("Login Attempts", 0, 10, 1)
trans_velocity = st.slider("Transaction Velocity", 0, 100, 1)
hour = st.slider("Hour of Transaction", 0, 23, 12)

# Prediction logic
if st.button("Predict Fraud"):
    # Define features
    numerical_features = ['amt', 'LoginAttempts', 'trans_velocity', 'hour']
    categorical_features = ['gender', 'city', 'state', 'Channel', 'Type of Card', 'Device Used', 'category', 'job']

    # Step 1: Create input DataFrame
    input_df = pd.DataFrame([{
        'gender': gender,
        'city': city,
        'state': state,
        'Channel': channel,
        'Type of Card': type_of_card,
        'Device Used': device,
        'category': category,
        'job': job,
        'amt': amt,
        'LoginAttempts': login_attempts,
        'trans_velocity': trans_velocity,
        'hour': hour
    }])

    # Step 2: Encode categorical features
    input_cat = input_df[categorical_features]
    input_encoded = encoder.transform(input_cat)
    encoded_cols = encoder.get_feature_names_out()
    encoded_df = pd.DataFrame(input_encoded, columns=encoded_cols, index=input_df.index)

    # Step 3: Scale numerical features
    input_num = input_df[numerical_features]
    numerical_scaled = scaler.transform(input_num)
    numerical_df = pd.DataFrame(numerical_scaled, columns=numerical_features, index=input_df.index)

    # Step 4: Align encoded_df to expected columns (fill missing with 0)
    for col in expected_columns:
        if col not in encoded_df.columns:
            encoded_df[col] = 0
    encoded_df = encoded_df[expected_columns]

    # Step 5: Combine scaled numerical + encoded categorical
    final_input = pd.concat([numerical_df, encoded_df], axis=1)

    # Step 6: Predict
    prediction = model.predict(final_input)[0]
    st.success("✅ Not Fraud" if prediction == 0 else "🚨 Fraud Detected!")
