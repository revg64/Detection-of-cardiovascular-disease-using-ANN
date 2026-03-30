import os
from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
from predict import predict_image

app = Flask(__name__)
app.secret_key = "secret123"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["MYSQL_HOST"] = "localhost"
app.config["MYSQL_USER"] = "root"
app.config["MYSQL_PASSWORD"] = "Rr9052251174."
app.config["MYSQL_DB"] = "ecg_users"

mysql = MySQL(app)

REASONS = {
    "Normal": "The ECG shows a well-defined P wave followed by a narrow QRS complex and a normal T wave.",
    "Myocardial Infarction": "The ECG exhibits abnormal Q waves and ST-segment deviations.",
    "History of MI": "The ECG shows persistent pathological Q waves.",
    "Abnormal Heartbeat": "The ECG demonstrates irregular P wave timing.",
}

SUGGESTIONS = {
    "Normal": "Lead a healthy lifestyle by being physically active, consuming a healthy diet, and getting regular health check-ups.",
    "Myocardial Infarction": "Immediately seek medical help, take medication as prescribed, avoid stressful situations, and eat a heart-healthy diet.",
    "History of MI": "Regular follow-up is strongly advised. Adhere to medication, monitor heart conditions, and lead a healthy lifestyle.",
    "Abnormal Heartbeat": "Get checked by a cardiologist for further evaluation. Limit caffeine consumption, avoid stressful situations, and monitor heart rhythm regularly."
}

@app.route("/", methods=["GET","POST"])
def login():

    error = None

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email=%s AND password=%s",(email,password))
        user = cur.fetchone()

        if user:
            session["user"] = user[1]
            return redirect(url_for("home"))
        else:
            error = "User does not exist or incorrect password"

    return render_template("login.html", error=error)


@app.route("/register", methods=["GET","POST"])
def register():

    if request.method == "POST":

        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        cur = mysql.connection.cursor()
        cur.execute(
            "INSERT INTO users(username,email,password) VALUES(%s,%s,%s)",
            (username,email,password)
        )
        mysql.connection.commit()

        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/home", methods=["GET","POST"])
def home():

    if "user" not in session:
        return redirect(url_for("login"))

    result = None
    reason = None
    suggestion = None

    if request.method == "POST":

        file = request.files.get("image")

        if file and file.filename:

            path = os.path.join(UPLOAD_FOLDER,file.filename)
            file.save(path)

            result = predict_image(path)
            reason = REASONS[result]
            suggestion = SUGGESTIONS[result]

    return render_template("index.html", result=result, reason=reason, suggestion=suggestion)


@app.route("/logout")
def logout():
    session.pop("user",None)
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True)