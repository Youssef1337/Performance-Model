from flask import Flask, request, render_template_string, flash
import os, logging, pickle
import pandas as pd
from datetime import datetime

# Initialize Flask app and logging
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'CHANGE_ME')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'student_data_realistic.csv')
FEATURES_PATH = os.path.join(BASE_DIR, '..', 'models', 'feature_list.txt')
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'random_forest_model.pkl')

# Load historical student data
try:
    student_data_df = pd.read_csv(DATA_PATH)
    logger.info(f'Loaded {len(student_data_df)} historical records from {DATA_PATH}')
except Exception as e:
    logger.error(f'Failed to load historical data: {e}')
    student_data_df = pd.DataFrame()

# Load features and model
try:
    with open(FEATURES_PATH) as f:
        FEATURES = [line.strip() for line in f]
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    logger.info('Model and features loaded successfully')
except Exception as e:
    logger.error(f'Error loading model or feature list: {e}')
    from sklearn.ensemble import RandomForestClassifier
    FEATURES = [
        'has_prerequisites', 'credits', 'attendance_rate',
        'midterm_score', 'final_score', 'assignment_score',
        'past_avg_total_score', 'past_avg_midterm_score', 'past_avg_final_score',
        'past_avg_assignment_score', 'past_avg_attendance', 'past_pass_rate',
        'courses_taken', 'dept_pass_rate', 'prereq_avg_score', 'prereq_pass_rate'
    ]
    model = RandomForestClassifier()
    logger.warning('Using dummy RandomForestClassifier and default features')

# Helper: extract student history features
def get_history_features(student_id, course_code=None):
    df = student_data_df
    recs = df[df['student_id'] == student_id]
    if recs.empty:
        return {k: 0 for k in FEATURES if k.startswith('past_') or k in ['courses_taken', 'dept_pass_rate', 'prereq_avg_score', 'prereq_pass_rate']}
    past = recs if course_code is None else recs[recs['course_code'] != course_code]
    if past.empty:
        past = recs
    feats = {
        'past_avg_total_score': past['total_score'].mean(),
        'past_avg_midterm_score': past['midterm_score'].mean(),
        'past_avg_final_score': past['final_score'].mean(),
        'past_avg_assignment_score': past['assignment_score'].mean(),
        'past_avg_attendance': past['attendance_rate'].mean(),
        'past_pass_rate': past['passed'].mean(),
        'courses_taken': len(past)
    }
    # Department pass rate
    if course_code:
        dept = course_code.split()[0]
        dept_past = past[past['course_code'].str.startswith(dept)]
        feats['dept_pass_rate'] = dept_past['passed'].mean() if not dept_past.empty else feats['past_pass_rate']
    else:
        feats['dept_pass_rate'] = feats['past_pass_rate']
    # Prerequisite features
    prereq_past = past[past['has_prerequisites']]
    feats['prereq_avg_score'] = prereq_past['total_score'].mean() if not prereq_past.empty else feats['past_avg_total_score']
    feats['prereq_pass_rate'] = prereq_past['passed'].mean() if not prereq_past.empty else feats['past_pass_rate']
    return feats

# Inline HTML template
HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Academic Performance Prediction</title>
<style>
  body { font-family: Arial, sans-serif; background: #f5f5f5; }
  .container { max-width: 600px; margin: 2rem auto; background: #fff; padding: 1.5rem; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.1); }
  label { display:block; margin-top:1rem; font-weight:bold; }
  input, select { width:100%; padding:0.5rem; margin-top:0.3rem; border:1px solid #ccc; border-radius:4px; }
  button { margin-top:1rem; padding:0.7rem 1.2rem; background:#3498db; color:#fff; border:none; cursor:pointer; border-radius:4px; }
  .result { margin-top:1.5rem; padding:1rem; background:#e8f8f5; border-left:4px solid #1abc9c; }
  .error { color:#c0392b; margin-top:1rem; }
</style>
</head>
<body>
<div class="container">
  <h1>Academic Performance Prediction</h1>
  {% with msgs = get_flashed_messages() %}{% if msgs %}<div class="error">{{ msgs[0] }}</div>{% endif %}{% endwith %}
  <form method="post">
    <label>Student ID</label><input type="number" name="student_id" required value="{{ request.form.student_id }}">
    <label>Course Code</label><input type="text" name="course_code" required value="{{ request.form.course_code }}">
    <label>Has Prerequisites?</label>
    <select name="has_prerequisites">
      <option value="1" {% if request.form.has_prerequisites=='1' %}selected{% endif %}>Yes</option>
      <option value="0" {% if request.form.has_prerequisites=='0' %}selected{% endif %}>No</option>
    </select>
    <label>Credits</label><input type="number" step="0.5" name="credits" min="1" max="6" value="{{ request.form.credits or 3 }}">
    <label>Attendance Rate (%)</label><input type="number" name="attendance" min="0" max="100" required value="{{ request.form.attendance or 80 }}">
    <label>Midterm Score</label><input type="number" name="midterm" min="0" max="100" value="{{ request.form.midterm }}">
    <label>Final Score</label><input type="number" name="final" min="0" max="100" value="{{ request.form.final }}">
    <label>Assignment Score</label><input type="number" name="assignment" min="0" max="100" value="{{ request.form.assignment }}">
    <button type="submit">Predict</button>
  </form>
  {% if prediction %}
    <div class="result">
      <h2>{{ 'Likely to Pass 🎉' if prediction=='Yes' else 'At Risk of Failing ⚠️' }}</h2>
      <p><strong>Probability:</strong> {{ '%.1f%%' % (probability*100) }}</p>
      <p><small>Prediction time: {{ date }}</small></p>
    </div>
  {% endif %}
</div>
</body>
</html>
'''

# Main route
@app.route('/', methods=['GET','POST'])
def index():
    prediction = None
    probability = None
    date = datetime.now().strftime('%Y-%m-%d %H:%M')
    if request.method == 'POST':
        try:
            sid = int(request.form['student_id'])
            code = request.form['course_code'].strip()
            pre = int(request.form['has_prerequisites'])
            cr = float(request.form['credits'])
            att = float(request.form['attendance'])
            
            # Extract historical features for this student
            hist = get_history_features(sid, code)
            
            # Use historical averages for missing scores instead of zero
            mid_input = request.form.get('midterm')
            fin_input = request.form.get('final')
            asm_input = request.form.get('assignment')
            
            # If scores are not provided, use historical averages
            mid = float(mid_input) if mid_input else hist['past_avg_midterm_score']
            fin = float(fin_input) if fin_input else hist['past_avg_final_score'] 
            asm = float(asm_input) if asm_input else hist['past_avg_assignment_score']
            
            # Log the scores being used
            logger.info(f"Scores for student {sid} - Midterm: {mid:.2f}, Final: {fin:.2f}, Assignment: {asm:.2f}")
            
            # Create data frame with features
            data = {
                'has_prerequisites': [pre], 
                'credits': [cr], 
                'attendance_rate': [att],
                'midterm_score': [mid], 
                'final_score': [fin], 
                'assignment_score': [asm],
                **{k: [v] for k, v in hist.items()}
            }
            
            # Create dataframe with only needed features in correct order
            df_in = pd.DataFrame(data)[FEATURES]
            
            # Make prediction
            proba = model.predict_proba(df_in)[:,1][0]
            prediction = 'Yes' if proba >= 0.5 else 'No'
            probability = proba
            
            # Log prediction details
            logger.info(f"PREDICTION: Student {sid}, Course {code}, Probability: {proba:.4f}, Result: {prediction}")
        except Exception as e:
            logger.error(f'Prediction error: {e}')
            flash('Error during prediction. Please check inputs.')
      # Render the template with results
    return render_template_string(HTML,
                                 prediction=prediction,
                                 probability=probability,
                                 date=date)

if __name__ == '__main__':
    app.run(debug=True)