function Q3=ProcessQ3()
[~, Q3, ~, fs]=ReadSound();

R_0 = 0;
R = zeros([length(Q3)-1,1]);
size(R)


for i = 0:(length(Q3)-1)
    for j = 1:(length(Q3)-i)
        if i == 0
            R_0 = R_0 + Q3(j)*Q3(j+i);
        else
            R(i) = R(i) + Q3(j)*Q3(j+i);
        end
    end
end
R = R ./ R_0;

n = linspace(1, length(Q3)-1, length(Q3)-1);
plot(n, R);

[r, T] = max(R(10:end), [], 1) ;
T = T + 9;
alpha = (1-sqrt(1-4*r^2))/(2*r);
disp(alpha);
for i = 1:(length(Q3)-T)
    Q3(i + T) = Q3(i + T) - alpha * Q3(i);
end

Q3_normalized = Q3 ./ abs(max([Q3;-Q3]));
audiowrite("./Q3_processed.wav", Q3_normalized, fs);