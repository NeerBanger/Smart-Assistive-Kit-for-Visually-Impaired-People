import pyttsx3
engine = pyttsx3.init()
sound = engine.getProperty('voices')
rate = engine.getProperty('rate')
volume = engine.getProperty('volume')
print ("Sound = ",sound)
print ("Rate = ",rate)
print ("Volume = ",volume)
engine.setProperty('rate' , rate-70)
engine.setProperty('voice' ,sound[1].id)
str = input("Enter the text : ")
#engine.say('{}'.format("There is a person in front of you. He is watching you."))
engine.say('{}'.format(str))
engine.runAndWait()