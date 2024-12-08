from flask import Flask, request, render_template, redirect, url_for, session
from models.slopeone_recommender import get_recommendations  # ajuste para o arquivo correto
from models.itemitem_recommender import recommend_for_user  # ajuste para o arquivo correto

app = Flask(__name__)
app.secret_key = "chave_secreta_segura"

# rota principal (index.html)
@app.route("/")
def index():
    return render_template("index.html")

# rota para login.html
@app.route("/login")
def login():
    return render_template("login.html")

# rota para processar o formulário de recomendações
@app.route("/recommendations", methods=["POST"])
def recommendations():
    user_id = request.form.get("user_id")
    algorithm_choice = request.form.get("algorithm")
    
    # validações básicas
    if not user_id or not algorithm_choice:
        return redirect(url_for("login"))
    
    # armazena o ID do usuário na sessão
    session["user_id"] = int(user_id)
    
    # redireciona para a rota específica do algoritmo escolhido
    if algorithm_choice == "slopeone":
        return redirect(url_for("slopeone_recommendations"))
    elif algorithm_choice == "itemitem":
        return redirect(url_for("itemitem_recommendations"))
    else:
        return redirect(url_for("login"))

# rota para recomendações
@app.route("/recommendations/slopeone", methods=["GET", "POST"])
def slopeone_recommendations():
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("login"))

    # obter recomendações
    slope_one_recs = get_recommendations(user_id)
    return render_template("slopeone_recommendations.html", slope_one_recs=slope_one_recs)

# rota para recomendações do algoritmo
@app.route("/recommendations/itemitem", methods=["GET", "POST"])
def itemitem_recommendations():
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("login"))

    # obter recomendações
    knn_recs = recommend_for_user(user_id)
    return render_template("itemitem_recommendations.html", knn_recs=knn_recs)

if __name__ == "__main__":
    app.run(debug=True)
