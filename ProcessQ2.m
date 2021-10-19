function Q2=ProcessQ2()
[Q2, ~, ~, fs]=ReadSound();

T = 1000;
alpha = 0.5;

for i = 1:(length(Q2)-T)
    Q2(i + T) = Q2(i + T) - alpha * Q2(i);
end

Q2_normalized = Q2 ./ abs(max([Q2;-Q2]));
audiowrite("./Q2_processed.wav", Q2_normalized, fs);