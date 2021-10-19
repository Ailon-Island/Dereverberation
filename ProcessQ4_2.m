function Q4=ProcessQ4_2()

[~, ~, Q4, fs]=ReadSound();

R = xcorr(Q4, 'normalized');
R = R(length(Q4):end);
size(R);

n = 1:(length(R));
plot(n, R(n));
 
[r, T] = sort(R, 'descend');
i = 2;
while i <= length(T)
    if T(i) > 500
        T0 = T(i) - 1;
        r0 = r(i);
        break;
    end
    i = i + 1;
end
while i <= length(T)
    if T(i) - T0 > 200
        T1 = T(i) - 1;
        r1 = r(i);
        break;
    end
    i = i + 1;
end

alpha0 = (r0-sqrt( r0^2 - 4*r0*r0*(r0^2+r1^2) ))/2/(r0^2+r1^2);
alpha1 = alpha0 * r1 / r0;

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