from testCNN import cnn_dronePred
from flask import Flask, render_template, request
import os, random

app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

@app.route("/")
def main():
    img_dir = os.listdir("static/images")
    randDrone = random.choice(img_dir)
    print(randDrone)

    result = cnn_dronePred(randDrone)
    
    if(result == "DRONE"):
        print("Drone present on property")

    else:
        print("No Drone on property")


    return render_template("index.html", d=randDrone, r=result)

if __name__ == "__main__":
    app.run(debug=True)
