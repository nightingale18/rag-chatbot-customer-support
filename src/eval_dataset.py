"""
Evaluation dataset.

Coverage:
  - doc_01  Vehicle Features
  - doc_02  Service Maintenance
  - doc_03  Warranty
  - doc_04  Ordering Process
  - doc_05  Electric Vehicles
  - doc_06  Customer Support

"""

EVAL_DATASET = [
    # ──────────────────────────────────────────────
    # doc_01_vehicle_features
    # ──────────────────────────────────────────────
    {
        "id": "feat_01",
        "document": "doc_01_vehicle_features.txt",
        "question": "What wheel sizes are available?",
        "reference": (
            "Alloy wheel designs range from 17 to 21 inches depending on the model."
        ),
    },
    {
        "id": "feat_02",
        "document": "doc_01_vehicle_features.txt",
        "question": "What upholstery options can I choose for the interior?",
        "reference": (
            "Upholstery options include fabric, synthetic leather, genuine leather, "
            "and premium leather."
        ),
    },
    {
        "id": "feat_03",
        "document": "doc_01_vehicle_features.txt",
        "question": "What technology packages are offered and can I use my phone as a car key?",
        "reference": (
            "Technology packages include a digital cockpit with navigation, Head-Up Display, "
            "parking assistant with surround-view cameras, and a digital car key that is "
            "compatible with smartphones."
        ),
    },
    {
        "id": "feat_04",
        "document": "doc_01_vehicle_features.txt",
        "question": "How many paint colours can I pick from?",
        "reference": (
            "There are over 15 standard paint colours, plus premium and matte finish options."
        ),
    },
    {
        "id": "feat_05",
        "document": "doc_01_vehicle_features.txt",
        "question": "Can I save my configuration and send it to a dealer?",
        "reference": (
            "Yes. Configurations can be saved and shared online. "
            "A completed configuration can be sent directly to a dealer for a personalised offer."
        ),
    },
    {
        "id": "feat_06",
        "document": "doc_01_vehicle_features.txt",
        "question": "Can I still change my chosen trim and colour after I have ordered the car?",
        "reference": (
            "Configuration changes are possible until the vehicle enters production. "
            "After that, changes can no longer be made."
        ),
    },
    # ──────────────────────────────────────────────
    # doc_02_service_maintenance
    # ──────────────────────────────────────────────
    {
        "id": "svc_01",
        "document": "doc_02_service_maintenance.txt",
        "question": "How often does the engine oil need to be changed?",
        "reference": (
            "Engine oil should be changed every 15,000 km or every 12 months, "
            "whichever comes first."
        ),
    },
    {
        "id": "svc_02",
        "document": "doc_02_service_maintenance.txt",
        "question": "When does the brake fluid need replacing?",
        "reference": (
            "Brake fluid must be replaced every 2 years, regardless of mileage. "
            "This applies to both combustion and electric vehicles."
        ),
    },
    {
        "id": "svc_03",
        "document": "doc_02_service_maintenance.txt",
        "question": "Does my electric car need oil changes?",
        "reference": (
            "No. Electric vehicles do not require engine oil changes. "
            "Service intervals are significantly longer due to fewer moving parts."
        ),
    },
    {
        "id": "svc_04",
        "document": "doc_02_service_maintenance.txt",
        "question": "What is the CBS system and where can I see the service reminder?",
        "reference": (
            "CBS stands for Condition Based Service. It monitors key components and "
            "calculates individual service intervals based on actual driving conditions. "
            "The remaining service life is displayed in the vehicle's infotainment system "
            "and in the companion mobile app."
        ),
    },
    {
        "id": "svc_05",
        "document": "doc_02_service_maintenance.txt",
        "question": "How frequently should the cabin air filter be replaced?",
        "reference": (
            "The cabin air filter should be replaced every 20,000 km or annually, "
            "whichever comes first."
        ),
    },
    {
        "id": "svc_06",
        "document": "doc_02_service_maintenance.txt",
        "question": "Are there prepaid service plans available and what do they cover?",
        "reference": (
            "Yes, pre-paid maintenance packages are available covering 3, 4, or 5 years."
        ),
    },
    {
        "id": "svc_07",
        "document": "doc_02_service_maintenance.txt",
        "question": "At what mileage interval should spark plugs be changed on petrol engines?",
        "reference": (
            "Spark plugs on petrol (gasoline) engines should be replaced every 60,000 km."
        ),
    },
    # ──────────────────────────────────────────────
    # doc_03_warranty
    # ──────────────────────────────────────────────
    {
        "id": "war_01",
        "document": "doc_03_warranty.txt",
        "question": "How long is the standard warranty?",
        "reference": ("The standard warranty lasts 2 years with unlimited mileage."),
    },
    {
        "id": "war_02",
        "document": "doc_03_warranty.txt",
        "question": "How long does the paintwork warranty last?",
        "reference": (
            "The paintwork warranty lasts 3 years and covers paint defects from the "
            "manufacturing process."
        ),
    },
    {
        "id": "war_03",
        "document": "doc_03_warranty.txt",
        "question": "What is the duration of the corrosion warranty?",
        "reference": (
            "The corrosion warranty lasts 12 years and covers perforation corrosion, "
            "which is rust that forms from the inside out."
        ),
    },
    {
        "id": "war_04",
        "document": "doc_03_warranty.txt",
        "question": "What exactly does the EV battery warranty cover and for how long?",
        "reference": (
            "The EV battery warranty lasts 8 years or 160,000 km, whichever comes first. "
            "It covers battery capacity falling below 70% of the original capacity, "
            "battery cell defects, and high-voltage components."
        ),
    },
    {
        "id": "war_05",
        "document": "doc_03_warranty.txt",
        "question": "Are brake pads and wiper blades covered under the standard warranty?",
        "reference": (
            "No. Normal wear items such as brake pads, tires, wiper blades, and bulbs "
            "are excluded from the standard warranty."
        ),
    },
    {
        "id": "war_06",
        "document": "doc_03_warranty.txt",
        "question": "If a stone chip damages my paint, is that covered by the warranty?",
        "reference": (
            "No. The paintwork warranty does not cover stone chips, scratches, "
            "or environmental damage. It only covers paint defects from the manufacturing process."
        ),
    },
    {
        "id": "war_07",
        "document": "doc_03_warranty.txt",
        "question": "Can I extend my warranty and when do I need to buy it?",
        "reference": (
            "Yes, an extended warranty is available for purchase before the standard warranty "
            "expires. Options are 1 or 2 additional years, and it can be purchased at any "
            "authorised dealer."
        ),
    },
    {
        "id": "war_08",
        "document": "doc_03_warranty.txt",
        "question": "Does the corrosion warranty require anything from me as the owner?",
        "reference": (
            "Yes. The corrosion warranty requires regular inspections at authorised service "
            "partners to remain valid."
        ),
    },
    {
        "id": "war_09",
        "document": "doc_03_warranty.txt",
        "question": "Where do I go to make a warranty claim?",
        "reference": (
            "All warranty claims must be processed through authorised service partners."
        ),
    },
    # ──────────────────────────────────────────────
    # doc_04_ordering_process
    # ──────────────────────────────────────────────
    {
        "id": "ord_01",
        "document": "doc_04_ordering_process.txt",
        "question": "How large is the deposit when placing a vehicle order?",
        "reference": ("The deposit is typically 10% of the vehicle price."),
    },
    {
        "id": "ord_02",
        "document": "doc_04_ordering_process.txt",
        "question": "How long does production usually take?",
        "reference": (
            "Production typically takes 6 to 10 weeks for standard configurations."
        ),
    },
    {
        "id": "ord_03",
        "document": "doc_04_ordering_process.txt",
        "question": "Can I track my car during production and how?",
        "reference": (
            "Yes. You can track the production status of your vehicle via the mobile app."
        ),
    },
    {
        "id": "ord_04",
        "document": "doc_04_ordering_process.txt",
        "question": "What are the steps involved in ordering a new vehicle?",
        "reference": (
            "The process has four steps: first configure the car online or at a dealer; "
            "then consult a dealer to discuss pricing and financing; then sign the purchase "
            "or lease agreement and pay the deposit; finally wait for production and collect "
            "the vehicle, either at the dealer or via a factory pickup experience."
        ),
    },
    {
        "id": "ord_05",
        "document": "doc_04_ordering_process.txt",
        "question": "Is it possible to modify my order after I have signed the agreement?",
        "reference": (
            "Order modifications are possible until the vehicle enters production. "
            "After production starts, changes can no longer be made."
        ),
    },
    {
        "id": "ord_06",
        "document": "doc_04_ordering_process.txt",
        "question": "What financing options do you offer?",
        "reference": (
            "Financing options include competitive financing rates, balloon financing with "
            "a guaranteed future value, and leasing for both private and business customers."
        ),
    },
    {
        "id": "ord_07",
        "document": "doc_04_ordering_process.txt",
        "question": "Do I have to go to a dealer or can I configure the car myself online?",
        "reference": (
            "You can configure the car yourself using the Online Configurator on the website, "
            "or visit a dealer directly. You can save your configuration with a code and then "
            "present it to a dealer."
        ),
    },
    # ──────────────────────────────────────────────
    # doc_05_electric_vehicles
    # ──────────────────────────────────────────────
    {
        "id": "ev_01",
        "document": "doc_05_electric_vehicles.txt",
        "question": "What is the maximum DC fast charging speed?",
        "reference": (
            "DC fast charging supports up to 200 kW, depending on the model."
        ),
    },
    {
        "id": "ev_02",
        "document": "doc_05_electric_vehicles.txt",
        "question": "How long does it take to charge from 10% to 80% at a fast charger?",
        "reference": (
            "Charging from 10% to 80% takes approximately 30 to 35 minutes at a fast charger."
        ),
    },
    {
        "id": "ev_03",
        "document": "doc_05_electric_vehicles.txt",
        "question": "How many public charging points can I access in Europe?",
        "reference": (
            "There are over 600,000 public charging points available across Europe."
        ),
    },
    {
        "id": "ev_04",
        "document": "doc_05_electric_vehicles.txt",
        "question": "What are my options for charging at home?",
        "reference": (
            "Home charging is available via a Wallbox in either 11 kW or 22 kW AC configurations."
        ),
    },
    {
        "id": "ev_05",
        "document": "doc_05_electric_vehicles.txt",
        "question": "What is battery pre-conditioning and is it available?",
        "reference": (
            "Battery pre-conditioning is a feature that brings the battery to its optimal "
            "temperature before fast charging. It is included as part of the latest generation "
            "electric drive technology."
        ),
    },
    {
        "id": "ev_06",
        "document": "doc_05_electric_vehicles.txt",
        "question": "What is the driving range of your electric models?",
        "reference": (
            "The electric vehicle lineup offers ranges from 400 to 630 km under WLTP conditions, "
            "covering both fully electric (BEV) and plug-in hybrid (PHEV) models."
        ),
    },
    {
        "id": "ev_07",
        "document": "doc_05_electric_vehicles.txt",
        "question": "Are there government incentives for buying an electric car?",
        "reference": (
            "Yes. Available incentives vary by country and include an environmental bonus, "
            "tax benefits for electric company cars, and reduced parking fees and toll exemptions "
            "in select cities."
        ),
    },
    {
        "id": "ev_08",
        "document": "doc_05_electric_vehicles.txt",
        "question": "Can I pay per charge or is there a subscription plan?",
        "reference": (
            "Both options are available. You can pay per use at competitive per-kWh rates, "
            "or choose a monthly subscription with reduced rates. Partnerships with major "
            "charging networks also offer preferential pricing."
        ),
    },
    # ──────────────────────────────────────────────
    # doc_06_customer_support
    # ──────────────────────────────────────────────
    {
        "id": "sup_01",
        "document": "doc_06_customer_support.txt",
        "question": "What are the customer hotline opening hours on Saturdays?",
        "reference": (
            "The customer hotline is available on Saturdays from 9:00 to 17:00."
        ),
    },
    {
        "id": "sup_02",
        "document": "doc_06_customer_support.txt",
        "question": "Is roadside assistance available on public holidays?",
        "reference": (
            "Yes. Roadside assistance is available 24 hours a day, 7 days a week, "
            "365 days a year, including public holidays."
        ),
    },
    {
        "id": "sup_03",
        "document": "doc_06_customer_support.txt",
        "question": "How do I activate roadside assistance if I break down?",
        "reference": (
            "Roadside assistance can be activated by phone, via the mobile app, "
            "or by pressing the SOS button inside the vehicle."
        ),
    },
    {
        "id": "sup_04",
        "document": "doc_06_customer_support.txt",
        "question": "How quickly will I get a reply if I send an email?",
        "reference": ("The email contact form has a response time of 24 to 48 hours."),
    },
    {
        "id": "sup_05",
        "document": "doc_06_customer_support.txt",
        "question": "What happens if my complaint is not resolved?",
        "reference": (
            "If a complaint is not resolved, a regional customer relations manager can be "
            "involved. If the issue remains unresolved, an ombudsman process is also available."
        ),
    },
    {
        "id": "sup_06",
        "document": "doc_06_customer_support.txt",
        "question": "Is roadside assistance included in my warranty or do I pay extra?",
        "reference": (
            "Roadside assistance is included for all vehicles within the warranty period "
            "at no extra cost."
        ),
    },
    {
        "id": "sup_07",
        "document": "doc_06_customer_support.txt",
        "question": "Can I book a service appointment through the app?",
        "reference": (
            "Yes. The mobile app supports in-app support and service booking."
        ),
    },
    {
        "id": "sup_08",
        "document": "doc_06_customer_support.txt",
        "question": "What services does roadside assistance actually provide?",
        "reference": (
            "Roadside assistance provides on-site repair, towing, and mobility services."
        ),
    },
    # ──────────────────────────────────────────────
    # Out-of-scope questions
    # ──────────────────────────────────────────────
    {
        "id": "oos_01",
        "document": None,
        "question": "What is the price of the base model?",
        "reference": (
            "The chatbot should say it does not have pricing information and direct "
            "the user to contact support or a dealer."
        ),
    },
    {
        "id": "oos_02",
        "document": None,
        "question": "Can I trade in my old car?",
        "reference": (
            "The chatbot should say it does not have information about trade-ins and "
            "direct the user to contact a dealer."
        ),
    },
    {
        "id": "oos_03",
        "document": None,
        "question": "Do you offer a test drive?",
        "reference": (
            "The chatbot should say it does not have information about test drives "
            "and suggest contacting a dealer."
        ),
    },
    {
        "id": "oos_04",
        "document": None,
        "question": "What is the 0 to 100 km/h acceleration time for the M3?",
        "reference": (
            "The chatbot should say it does not have performance specification data "
            "and direct the user to the product website or a dealer."
        ),
    },
]


if __name__ == "__main__":
    from collections import Counter

    docs = Counter(e["document"] for e in EVAL_DATASET)

    print(f"Total samples : {len(EVAL_DATASET)}\n")

    print("\nBy source document:")
    for d, n in sorted(docs.items(), key=lambda x: (x[0] is None, x[0])):
        label = d if d else "(out-of-scope)"
        print(f"  {label:<40} {n}")
