import streamlit as st
import pandas as pd
import joblib

# Load the model and scaler
ensemble_model = joblib.load('ensemble_model.pkl')
scaler = joblib.load('scaler.pkl')
frequency_encoding_map = joblib.load('frequency_encoding_map.pkl')

# Function to make predictions
def predict(data):
    for col in frequency_encoding_map:
        data[col] = data[col].map(frequency_encoding_map[col]).fillna(0)
    
    numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
    data[numerical_columns] = scaler.transform(data[numerical_columns])
    prediction = ensemble_model.predict(data)
    return prediction[0]

# CSS for styling
st.markdown("""
<style>
.title {
    font-size: 2em;
    color: #000000;
    text-align: center;
    margin-bottom: 20px;
}
.subheader {
    font-size: 1.5em;
    color: #000000;
}
.input-field {
    margin-bottom: 15px;
}
.button {
    background-color: #4CAF50;
    color: white;
    padding: 10px 20px;
    border: none;
    cursor: pointer;
}
.button:hover {
    background-color: #45a049;
}
</style>
""", unsafe_allow_html=True)

# Streamlit app layout
st.markdown('<div class="title">Student Performance Prediction</div>', unsafe_allow_html=True)

st.subheader("Enter Student Details")
input_columns = st.columns(2)

# Input fields for user data
school = st.selectbox("School (GP=Public School, MS= Private School)", ['GP', 'MS'], key='school', help='Select the type of school')
sex = st.selectbox("Sex", ['F', 'M'], key='sex')
age = st.number_input("Age", min_value=10, max_value=20, value=17, key='age', help='Enter the age of the student')
address = st.selectbox("Address (U= Urban,R= Rural)", ['U', 'R'], key='address')
famsize = st.selectbox("Family Size (GT3= Greater than 3 , LE3 = Lesser than 3)", ['GT3', 'LE3'], key='famsize')
Pstatus = st.selectbox("Parental Status (T= Together, A= Apart)", ['A', 'T'], key='Pstatus')
Medu = st.number_input("Mother's Education (0-4)", min_value=0, max_value=4, value=4, key='Medu')
Fedu = st.number_input("Father's Education (0-4)", min_value=0, max_value=4, value=4, key='Fedu')
Mjob = st.selectbox("Mother's Job", ['health', 'services', 'at_home', 'teacher', 'other'], key='Mjob')
Fjob = st.selectbox("Father's Job", ['health', 'services', 'at_home', 'teacher', 'other'], key='Fjob')
reason = st.selectbox("Reason for Choosing School", ['course', 'home', 'reputation', 'other'], key='reason')
guardian = st.selectbox("Guardian", ['mother', 'father', 'other'], key='guardian')
traveltime = st.number_input("Travel Time (1-4)", min_value=1, max_value=4, value=1, key='traveltime')
studytime = st.number_input("Study Time (1-4)", min_value=1, max_value=4, value=2, key='studytime')
failures = st.number_input("Number of Past Class Failures (1-4)", min_value=0, max_value=4, value=0, key='failures')
schoolsup = st.selectbox("School Support", ['yes', 'no'], key='schoolsup')
famsup = st.selectbox("Family Support", ['yes', 'no'], key='famsup')
paid = st.selectbox("Extra Paid Classes", ['yes', 'no'], key='paid')
activities = st.selectbox("Extracurricular Activities", ['yes', 'no'], key='activities')
nursery = st.selectbox("Attended Nursery", ['yes', 'no'], key='nursery')
higher = st.selectbox("Wants to Take Higher Education", ['yes', 'no'], key='higher')
internet = st.selectbox("Internet Access", ['yes', 'no'], key='internet')
romantic = st.selectbox("With a Romantic Partner", ['yes', 'no'], key='romantic')
famrel = st.number_input("Family Relationship Quality (1-4)", min_value=1, max_value=5, value=4, key='famrel')
freetime = st.number_input("Free Time After School (1-4)", min_value=1, max_value=5, value=3, key='freetime')
goout = st.number_input("Going Out with Friends (1-4)", min_value=1, max_value=5, value=4, key='goout')
Dalc = st.number_input("Dalc (Workday screen Consumption)(1-4)", min_value=1, max_value=5, value=1, key='Dalc')
Walc = st.number_input("Walc (Weekend screen Consumption)(1-4)", min_value=1, max_value=5, value=1, key='Walc')
health = st.number_input("Health Status(1-4)", min_value=1, max_value=5, value=3, key='health')
absences = st.number_input("Number of School Absences", min_value=0, max_value=100, value=0, key='absences')
Langauge = st.number_input("Language Score", min_value=0, max_value=100, value=100, key='Langauge')
Maths = st.number_input("Math Score", min_value=0, max_value=100, value=100, key='Maths')
Science = st.number_input("Science Score", min_value=0, max_value=100, value=100, key='Science')

# Create a DataFrame from the inputs
data = pd.DataFrame({
    'school': [school],
    'sex': [sex],
    'age': [age],
    'address': [address],
    'famsize': [famsize],
    'Pstatus': [Pstatus],
    'Medu': [Medu],
    'Fedu': [Fedu],
    'Mjob': [Mjob],
    'Fjob': [Fjob],
    'reason': [reason],
    'guardian': [guardian],
    'traveltime': [traveltime],
    'studytime': [studytime],
    'failures': [failures],
    'schoolsup': [schoolsup],
    'famsup': [famsup],
    'paid': [paid],
    'activities': [activities],
    'nursery': [nursery],
    'higher': [higher],
    'internet': [internet],
    'romantic': [romantic],
    'famrel': [famrel],
    'freetime': [freetime],
    'goout': [goout],
    'Dalc': [Dalc],
    'Walc': [Walc],
    'health': [health],
    'absences': [absences],
    'Langauge': [Langauge],
    'Maths': [Maths],
    'Science': [Science],
})

if st.button("LET's GO", key='submit', help='Click to make a prediction'):
    prediction = predict(data)
    st.success(f"The predicted percentage is: **{prediction:.2f}%**")
    
# Add more styling or modify existing elements as needed

    if prediction < 10:
        st.warning("The student's performance indicates a need for significant improvement, "
    "likely due to challenges in understanding the material or external factors. "
    "It's crucial to address these early to prevent further decline. "
    "Reviewing input data and providing support can help identify and resolve weaknesses. "
    "Assessing learning styles, access to resources, and personal challenges may offer insight.\n\n"
    "Suggestions for improvement:\n"
    "- Review study materials for comprehension, using tools like videos or quizzes.\n"
    "- Seek tutoring or additional support through mentoring or online platforms.\n"
    "- Create a structured study schedule to manage time efficiently and stay disciplined.\n"
    "- Practice active learning, such as summarizing and self-quizzing, to reinforce retention.\n"
    "- Monitor progress regularly and adjust study strategies if necessary.\n"
    "- Encourage a positive mindset, focusing on effort and resilience.\n"
    "- Balance academics with well-being, ensuring the student maintains a healthy lifestyle for better concentration.")
    elif prediction < 25:
        st.warning( "This indicates below-average performance. It's crucial to address these gaps promptly "
    "to prevent further decline. Identifying the key areas of weakness is the first step towards "
    "improvement. Often, difficulties in specific subjects or topics hinder overall performance, "
    "so pinpointing these challenges will help focus efforts where they are needed most.\n\n"
    "Suggestions for improvement:\n"
    "- Identify specific subjects or topics that are proving difficult and allocate extra time "
    "to study them.\n"
    "- Develop effective study habits, such as active learning techniques. This includes summarizing "
    "information, self-testing, and teaching concepts to others to deepen understanding.\n"
    "- Break down study sessions into manageable parts, ensuring consistent progress without burnout.\n"
    "- Utilize school resources like study groups or after-school tutoring. These resources provide "
    "personalized attention and support that can clarify challenging topics.\n"
    "- Establish a structured routine that balances study time with breaks, ensuring discipline "
    "while avoiding exhaustion.\n"
    "- Regularly monitor progress through quizzes or self-assessments to stay on track.")
    elif prediction < 35:
        st.info("This suggests average performance, which means there's room for improvement and growth. "
    "While the student may be meeting basic expectations, there are opportunities to elevate performance "
    "by implementing small, targeted strategies that enhance understanding and consistency.\n\n"
    "Suggestions for improvement:\n"
    "- Set achievable short-term academic goals to boost confidence. Breaking down larger academic targets "
    "into smaller milestones helps maintain motivation and a sense of progress.\n"
    "- Incorporate regular feedback sessions with teachers or mentors. These sessions can provide valuable insights "
    "into strengths and areas for improvement, allowing the student to adjust their study strategies accordingly.\n"
    "- Actively seek constructive feedback after assessments to identify patterns and gaps in learning.\n"
    "- Use online resources and tools for additional practice, especially in challenging areas. Platforms such as "
    "Khan Academy, Quizlet, or other educational websites can provide extra exercises to reinforce key concepts.\n"
    "- Build a study routine that integrates these tools with daily learning to maintain steady improvement.\n"
    "- Balance academic work with relaxation to avoid burnout and maintain a healthy study-life balance.")
    elif prediction < 45:
        st.info("This indicates slightly above-average performance. While the student is performing well, it's important "
    "to build on this foundation to achieve even higher academic success. There is potential for growth with "
    "a few strategic improvements to enhance understanding and consistency.\n\n"
    "Suggestions for improvement:\n"
    "- Engage in peer study sessions to enhance understanding. Collaborative learning can provide new perspectives "
    "on difficult topics and deepen comprehension by explaining concepts to others.\n"
    "- Review past assessments to identify areas for improvement. Analyzing mistakes made in quizzes, tests, or assignments "
    "can provide insight into specific weak points that need attention.\n"
    "- Establish a consistent homework and revision routine. Regular practice and review are key to reinforcing learning, "
    "preventing last-minute cramming, and ensuring long-term retention of information.\n"
    "- Break down study sessions into focused periods, balancing time between subjects and areas of strength and weakness.\n"
    "- Consider setting specific goals for each subject to further boost performance and track progress over time.\n"
    "- Stay proactive by seeking extra resources or practice opportunities to stay ahead and sharpen skills.")
    elif prediction < 50:
        st.success("This suggests good performance, but there's still potential for growth. The student has demonstrated a solid "
    "understanding of the material, but further improvement can be achieved by taking on new challenges and expanding "
    "beyond the current scope.\n\n"
    "Suggestions for improvement:\n"
    "- Explore advanced topics to deepen understanding. Delving into more complex areas of the subject matter will help "
    "solidify core knowledge while preparing the student for future learning. Engaging with advanced materials can also "
    "stimulate curiosity and critical thinking.\n"
    "- Participate in extracurricular activities that enhance learning. Joining academic clubs, competitions, or relevant "
    "projects can provide hands-on experience and practical application of what is being learned in the classroom.\n"
    "- Seek constructive feedback to identify further areas for growth. Proactively asking teachers, mentors, or peers for "
    "feedback after assessments or projects will highlight specific opportunities for improvement and guide focused efforts.\n"
    "- Set higher academic goals to continuously challenge personal limits and maintain momentum.\n"
    "- Use online courses or workshops to gain insights into advanced concepts or emerging topics in the field.")
    elif prediction < 65:
        st.success("This indicates very good performance. The student has shown consistent understanding and progress, but maintaining "
    "this level will require continued dedication and focus. Building on existing strengths while setting new goals will "
    "help sustain and elevate academic success.\n\n"
    "Suggestions for improvement:\n"
    "- Set personal challenges to strive for higher grades. By aiming for higher targets, the student can continue pushing "
    "their academic boundaries and avoid complacency.\n"
    "- Explore resources for independent learning, such as books or online courses. Independent study fosters self-motivation "
    "and a deeper, broader understanding of subjects beyond the classroom curriculum.\n"
    "- Consider joining clubs or teams that promote collaborative learning. Engaging with peers in academic clubs, debate teams, "
    "or study groups can enhance critical thinking, teamwork, and expose the student to diverse perspectives.\n"
    "- Stay curious by exploring interdisciplinary subjects or tackling new projects that apply classroom learning to real-world challenges.\n"
    "- Maintain a well-structured routine to ensure consistent progress without burning out, balancing academics with extracurricular activities.")
    elif prediction < 75:
        st.success("This suggests excellent performance, which is commendable. The student has clearly excelled and demonstrated a deep "
    "understanding of the material. To maintain this high standard, it's important to continue current practices while "
    "seeking opportunities for further development.\n\n"
    "Suggestions for improvement:\n"
    "- Maintain current study habits while seeking new learning opportunities. Building on strong routines will ensure "
    "consistent success, while exploring new topics or advanced materials will foster continuous growth.\n"
    "- Explore leadership roles in study groups or projects. Taking on leadership positions can enhance communication skills, "
    "boost confidence, and contribute to a stronger grasp of the subject by guiding others through complex ideas.\n"
    "- Engage in discussions to deepen understanding of complex topics. Participating in debates, academic forums, or group discussions "
    "can challenge the student to think critically, consider different viewpoints, and solidify their own understanding.\n"
    "- Continue setting ambitious goals to remain motivated and engaged with learning, while balancing academic rigor with personal development.")
    elif prediction < 85:
        st.success("This indicates outstanding performance, a great achievement. The student has consistently excelled and demonstrated "
    "an exceptional understanding of the material. To continue this trajectory of success, it's important to seek new challenges "
    "and opportunities for growth.\n\n"
    "Suggestions for improvement:\n"
    "- Challenge yourself with higher-level coursework or subjects. Engaging with advanced materials will not only broaden "
    "knowledge but also keep the learning experience stimulating and rewarding.\n"
    "- Mentor peers who may struggle, reinforcing your own knowledge. Teaching others is a powerful way to deepen understanding "
    "and solidify concepts, while also contributing positively to the learning community.\n"
    "- Set specific goals for each subject to maintain and enhance performance. Clear, achievable targets can provide motivation "
    "and a roadmap for continuous improvement, ensuring that even high achievers continue to grow.\n"
    "- Explore opportunities for research or independent projects that align with personal interests to further engage and enrich "
    "the academic experience.")
    elif prediction < 90:
        st.success("This suggests exceptional performance. You are on the right track towards excellence. Your dedication and effort have "
    "resulted in a strong understanding of the material, positioning you well for future success. To further enhance your "
    "academic journey, consider the following suggestions.\n\n"
    "Suggestions for improvement:\n"
    "- Pursue specialized areas of interest or advanced topics. Delving deeper into specific subjects can enrich your knowledge "
    "and maintain your engagement with learning.\n"
    "- Participate in competitions or projects to apply knowledge. Engaging in real-world challenges will help reinforce skills "
    "and provide practical experience that complements classroom learning.\n"
    "- Network with teachers or professionals for insights into advanced studies. Building relationships with mentors can open "
    "doors to opportunities and provide valuable guidance for your academic and career path.\n"
    "- Consider exploring interdisciplinary subjects that combine your interests, fostering a more holistic understanding of various fields.")
    elif prediction < 95:
        st.success("This suggests exceptional performance. You are excelling in your studies and setting a high standard for yourself and "
    "your peers. Your hard work and dedication are commendable, and to build on this success, consider the following suggestions.\n\n"
    "Suggestions for improvement:\n"
    "- Explore leadership opportunities in academic settings. Taking on leadership roles can enhance your skills, build confidence, "
    "and inspire others to achieve their best.\n"
    "- Contribute to peer tutoring programs to share knowledge. Mentoring fellow students not only reinforces your understanding but "
    "also fosters a supportive learning environment.\n"
    "- Set long-term academic and career goals to maintain motivation. Establishing clear, achievable objectives will help you stay focused "
    "and inspired as you continue your educational journey.\n"
    "- Engage in extracurricular activities that align with your interests, providing opportunities for growth beyond academics.")
    else:
        st.success("This suggests exceptional performance.")
    st.info("These predictions are based on the provided input data. Please review and consider additional resources if needed.")
