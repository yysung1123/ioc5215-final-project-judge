from flask import *
from werkzeug.utils import secure_filename
from nctu_oauth import Oauth, NCTU_OAUTH_URL, NYCU_OAUTH_URL
from predictor import check_range, predict
from enum import IntEnum
from datetime import datetime
from config import (SECRET_KEY, UPLOAD_FOLDER, NCTU_APP_REDIRECT_URI,
                    NCTU_APP_CLIENT_ID, NCTU_APP_CLIENT_SECRET, NYCU_APP_REDIRECT_URI,
                    NYCU_APP_CLIENT_ID, NYCU_APP_CLIENT_SECRET, DATABASE, TIME_VALID_UPPER_BOUND)
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import numpy as np
import os
import sqlite3

ALLOWED_EXTENSIONS = {'npy'}
PAGINATION = 50

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["10 per second"]
)

nctu = Oauth(
    redirect_uri=NCTU_APP_REDIRECT_URI,
    client_id=NCTU_APP_CLIENT_ID,
    client_secret=NCTU_APP_CLIENT_SECRET,
    oauth_url=NCTU_OAUTH_URL
)

nycu = Oauth(
    redirect_uri=NYCU_APP_REDIRECT_URI,
    client_id=NYCU_APP_CLIENT_ID,
    client_secret=NYCU_APP_CLIENT_SECRET,
    oauth_url=NYCU_OAUTH_URL
)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def init_db():
    with app.app_context():
        db = get_db()
        with app.open_resource('schema.sql', mode='r') as f:
            db.cursor().executescript(f.read())
        db.commit()


def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)

    return db


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


class SubmissionStatus(IntEnum):
    VALID = 0
    INVALID = 1
    ERROR = 2


app.add_template_global(SubmissionStatus, 'SubmissionStatus')


class Submission:
    def __init__(self, id, username, status, acc, time, error_msg, create_timestamp):
        self.id = id
        self.username = username
        self.status = SubmissionStatus(status)
        self.acc = acc
        self.time = time
        self.error_msg = error_msg

        if isinstance(create_timestamp, str):
            self.create_timestamp = datetime.strptime(
                create_timestamp, '%Y-%m-%d %H:%M:%S.%f')
        else:
            self.create_timestamp = create_timestamp

    @staticmethod
    def create(username, status, acc, time, error_msg):
        with app.app_context():
            db = get_db()

            create_timestamp = datetime.now()
            cur = db.cursor()
            cur.execute('INSERT INTO submissions (username, status, acc, time, error_msg, create_timestamp) VALUES (?, ?, ?, ?, ?, ?)',
                        [username, status, acc, time, error_msg, create_timestamp])
            db.commit()

            return Submission(cur.lastrowid, username, status, acc, time, error_msg, create_timestamp)

    @staticmethod
    def get_by_username(username, offset=0, limit=PAGINATION):
        with app.app_context():
            db = get_db()

            cur = db.cursor()
            cur.execute(
                'SELECT * FROM submissions WHERE username = ? ORDER BY create_timestamp DESC LIMIT ? OFFSET ?', [username, limit, offset])
            return [Submission(*row) for row in cur.fetchall()]

    @staticmethod
    def get_username_count(username):
        with app.app_context():
            db = get_db()

            cur = db.cursor()
            cur.execute(
                'SELECT COUNT(*) FROM submissions WHERE username = ?', [username])
            return cur.fetchone()[0]

    @staticmethod
    def get_leaderboard(offset=0, limit=PAGINATION):
        with app.app_context():
            db = get_db()

            cur = db.cursor()
            cur.execute('SELECT * FROM submissions WHERE status = ? ORDER BY acc DESC, create_timestamp LIMIT ? OFFSET ?',
                        [SubmissionStatus.VALID, limit, offset])
            return [Submission(*row) for row in cur.fetchall()]

    @staticmethod
    def get_leaderboard_count():
        with app.app_context():
            db = get_db()

            cur = db.cursor()
            cur.execute('SELECT COUNT(*) FROM submissions WHERE status = ?',
                        [SubmissionStatus.VALID])
            return cur.fetchone()[0]

    @staticmethod
    def get_teambest():
        with app.app_context():
            db = get_db()

            cur = db.cursor()
            cur.execute('SELECT *, MAX(acc) FROM submissions WHERE status = ? GROUP BY username ORDER BY acc DESC, create_timestamp',
                        [SubmissionStatus.VALID])
            return [Submission(*row[:-1]) for row in cur.fetchall()]


@app.route('/login')
def login():
    return render_template('login.html', session=session)


@app.route('/login/nctu')
def login_nctu():
    return nctu.authorize()


@app.route('/login/nycu')
def login_nycu():
    return nycu.authorize()


@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')


@app.route('/auth/nctu')
def auth_nctu():
    code = request.args.get('code')
    if code:
        if nctu.get_token(code):
            return redirect('/')

    return redirect('/login')


@app.route('/auth/nycu')
def auth_nycu():
    code = request.args.get('code')
    if code:
        if nycu.get_token(code):
            return redirect('/')

    return redirect('/login')


@app.route('/')
def index():
    return render_template('index.html', session=session)


@app.route('/nas', methods=['GET'])
def nas():
    return render_template('nas.html', session=session)


@app.route('/nas', methods=['POST'])
@limiter.limit("1 per second")
def submit():
    if not session.get('logged_in'):
        return redirect('/login')
    username = session.get('username')

    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if not file or not allowed_file(file.filename):
        flash("Not allowed extension (.npy)")
        return redirect(request.url)

    try:
        model_architecture = np.load(file)
    except Exception as e:
        print(e)
        flash("Not a valid npy file")
        return redirect(request.url)

    error, msgs = check_range(model_architecture)
    if error:
        Submission.create(username, SubmissionStatus.ERROR,
                          0.0, 0.0, "\n".join(msgs))
        return redirect('/submissions')

    acc, time = predict(model_architecture)

    if time <= TIME_VALID_UPPER_BOUND:
        submission = Submission.create(
            username, SubmissionStatus.VALID, acc, time, "")
    else:
        submission = Submission.create(
            username, SubmissionStatus.INVALID, acc, time, f"Time > {TIME_VALID_UPPER_BOUND}ms")

    file.save(os.path.join(
        app.config['UPLOAD_FOLDER'], f'{submission.id}.npy'))

    return redirect('/submissions')


@app.route('/leaderboard')
def leaderboard():
    offset = request.args.get('offset', default=0, type=int)
    leaderboard = Submission.get_leaderboard(offset=offset)
    count = Submission.get_leaderboard_count()
    teambest = Submission.get_teambest()
    return render_template('leaderboard.html', session=session,
                           leaderboard=leaderboard, offset=offset,
                           paginate=list(range(0, count, PAGINATION)), teambest=teambest)


@app.route('/submissions')
def submissions():
    if session.get('logged_in'):
        username = session.get('username')
        offset = request.args.get('offset', default=0, type=int)
        submissions = Submission.get_by_username(username, offset=offset)
        count = Submission.get_username_count(username)
        return render_template('submissions.html', session=session, submissions=submissions, offset=offset, paginate=list(range(0, count, PAGINATION)))

    return redirect('/login')
