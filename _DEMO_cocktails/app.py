from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def hello():
    image_url = " "
    if request.method == "POST":
        image_url = request.form["image_url"]
    return render_template("index.html", name=image_url)


'''@app.route("/sub", methods=["POST"])
def submit():
    image_url = ""
    if request.method == "POST":
        image_url = request.form["image_url"]
    return render_template("sub.html", name=image_url)'''


if __name__ == "__main__":
    app.run(debug=True)
