from whatsapp_api_client_python import API
greenAPI = API.GreenAPI("7105197606", "24f0fd5b279c45a28f1e2d38f8d1ec0dbcf26a571ea4482896")

def send_whatsapp_alert(result, phone_number):
    
    warning_threshold = 0.56  
    evacuation_threshold = 0.70
      
    if result >= evacuation_threshold:
        message = '''🚨 URGENT EVACUATION ALERT
⚠️ FLOOD RISK IS EXTREMELY HIGH!

Your area is at severe risk of flooding. Evacuate immediately to higher ground or a safe location.

1️⃣ Carry essentials (ID, cash, meds, food, water).
2️⃣ Turn off electricity and gas before leaving.
3️⃣ Avoid floodwaters—stay safe from strong currents.

DO NOT DELAY! ACT NOW!
For help, contact local authorities. Stay informed and stay safe!'''
        
        response = greenAPI.sending.sendMessage(phone_number, message)
        print(f"Evacuation Alert Sent: {response.data}")
    
    elif result >= warning_threshold:
        message = '''⚠️ FLOOD WARNING
There is a significant chance of flooding in your area. Stay alert and take these precautions:
git
1️⃣ Move to higher ground if you are in a low-lying area.
2️⃣ Prepare an emergency kit with essentials (ID, cash, meds, food, water).
3️⃣ Avoid walking or driving through floodwaters—stay safe from currents and debris.
4️⃣ Stay updated via official channels for further instructions.

Be prepared and stay safe!'''
        response = greenAPI.sending.sendMessage(phone_number, message)
        print(f"Flood Warning Sent: {response.data}")
    
    else:
        print("No alert sent. Flood probability is below the warning threshold.")




# if __name__ == "__main__":
#     flood_probability = 0.75
#     recipient_phone_number = "918851624048@c.us" 
    
#     # Send alert based on the prediction result
#     send_whatsapp_alert(flood_probability, recipient_phone_number)
