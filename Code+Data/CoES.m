%%% 计算CoES风险协变量

[T,N] = size(Daily_Return_Rate);
h = 242; % Roll_window
p = 0.05; 
CoES_time=zeros(N,N,T-h);
for t = 1:T-h 
    % CoVaR = zeros(N,N); 
    CoES = zeros(N,N);
    data = Daily_Return_Rate(t:t+h-1,:); 
    for i = 1:N
        for j = 1:N

            VaR_j=quantile(data(:,j),p); 
            index_ij = data(:,j)<VaR_j; 
            % CoVaR(i,j)=quantile(Data(index_ij,i),p);
            CoES(i,j)=mean(data(index_ij,i));
        end
    end
    % CoES = [CoVaR,CoES];
    CoES_time(:,:,t)=CoES;

end

Cosine_Similarity=zeros(N,N,(T-h));

for t=1:(T-h)
    for i=1:N
        X_it= CoES_time(i,:,t);
        Centered_X_it= X_it-mean(X_it);   % 去中心化

        for j=(i+1):N
            X_jt = CoES_time(:,j,t);
            Centered_X_jt= X_jt-mean(X_jt);   % 去中心化
            % 计算余弦相似度
            Cosine_Similarity(i,j,t) = dot(Centered_X_it,Centered_X_jt)/(norm(Centered_X_it)*norm(Centered_X_jt));
            Cosine_Similarity(j,i,t) = Cosine_Similarity(i,j,t); 
            % Connect(t,1)=sum(sum(Cosine_Similarity(:,:,t)));  
        end
    end
end

