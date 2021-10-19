%function Q4=ProcessQ4()
[~, ~, Q4, fs]=ReadSound();
Q30 = ProcessQ3();
Q30(length(Q4)) = 0;
dQ4 = Q4 - Q30;
length(dQ4)

R = xcorr(Q30, dQ4);
size(R);

n = 4000:(length(R)-4000);
plot(n, R(n));
 
[r, T] = sort(R, 'descend');
T0 = T(1);
r0 = r(1);
for i = 2:length(T)
    if abs(T(i) - T0) > 200
        T1 = T(i);
        r1 = r(i);
        break
    end
end
T0 = length(dQ4) - T0;
T1 = length(dQ4) - T1;

R0 = sum(Q30 .^ 2);

alpha0 = r0 / R0;
alpha1 = r1 / R0;

for i = 1:length(Q4)
    if i+T0 <= length(Q4)
        Q4(i + T0) = Q4(i + T0) - alpha0 * Q4(i);
    end
    
    if i+T1 <= length(Q4)
        Q4(i + T1) = Q4(i + T1) - alpha1 * Q4(i);
    end
end

Q4_normalized = Q4 ./ abs(max([Q4;-Q4]));
audiowrite("./Q3_processed.wav", Q4_normalized, fs);