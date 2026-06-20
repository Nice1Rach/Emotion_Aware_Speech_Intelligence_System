import pyttsx3

engine = pyttsx3.init('sapi5')

print("\nAVAILABLE VOICES\n")

for voice in engine.getProperty('voices'):
    print("Name:", voice.name)
    print("ID:", voice.id)
    print("-" * 50)