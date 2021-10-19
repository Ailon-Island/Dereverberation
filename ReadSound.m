function  [Q2, Q3, Q4, fs]=ReadSound()
fs=8192;

Q = load('-mat', "lab1.mat");
Q2 = Q.Q2;
Q3 = Q.Q3;
Q4 = Q.Q4;

Q2_normalized = Q2 ./ abs(max([Q2;-Q2]));
Q3_normalized = Q3 ./ abs(max([Q3;-Q3]));
Q4_normalized = Q4 ./ abs(max([Q4;-Q4]));

audiowrite("./Q2.wav" ,Q2_normalized, fs);
audiowrite("./Q3.wav" ,Q3_normalized, fs);
audiowrite("./Q4.wav" ,Q4_normalized, fs);
    



