a2=a(:,2);
k=30;                     
maxgen=100;               
w=0.8;                   
a3=3;
b3=5;
a_pbest=.000035 * k;
b_pbest=.5;
sizepop=15;               
% Vmax = 100000;Vmin = -100000;        
% popmax = 100000;popmin = -100000;   
% popmax = 21194;popmin = 824;
ap=0.05;                   
trace=zeros(maxgen,1);    
V=zeros(sizepop,k);   
ksort=sortrows(b,-2); 
ksort_all_id=ksort(:,1); 
ksort_k=ksort(1:k,:); 
ksort_select_id=ksort_k(:,1);
suiji=rand(k,1);       
X=zeros(sizepop,k);   
for j=1:sizepop
  for i=1:k
     if suiji(i)>0.5
         res=setdiff(ksort_all_id, ksort_select_id); 
         ksort_select_id(i)=res(randperm(numel(res),1));
     end
  end
X(j,:)=ksort_select_id;
end
[bestfitness,bestindex]=max(fitness);
Gbest=X(bestindex,:);     
fitnessGbest=bestfitness;   
Pbest=X;
fitnessPbest=fitness;       
for i=1:maxgen    
    for j=1:sizepop    
        [xpbest,ipx]=intersect(Pbest(j,:),X(j,:));
        v_pbest=ones(1, k);
        v_pbest(ipx)=0;
        [xgbest,igx]=intersect(Gbest,X(j,:));
        v_gbest=ones(1, k);
        v_gbest(igx)=0;
        k_pbest=1-length(ipx)/(2*k-length(ipx));
        exp_pbest=1 + exp(-a_pbest*k_pbest);
        c1=b_pbest/exp_pbest;
        k_gbest=1-length(igx)/(2*k-length(igx));
        exp_gbest=1+exp(-a_pbest*k_gbest);
        c2=b_pbest/exp_gbest;
        h(j,:)=w*V(j,:)+c1*rand*v_pbest+c2*rand*v_gbest; 
        exp_zhishu=-(a3*h(j,:)+ones(1,k)*b3);
        exp1=exp(exp_zhishu);
        H=1./(ones(1,k)+exp1);
        H_random=rand(1,k);
        for p=1:k
            if  H_random(p)>=H(p)
                h(j,p)=1;
            else
                h(j,p)=0;
            end
        end
%         V(j,:) = w*V(j,:) + c1*rand*(Pbest(j,:) - X(j,:)) + ... 
%             c2*rand*(Gbest - X(j,:));
%         V(j,V(j,:)>Vmax)=Vmax;
%         V(j,V(j,:)<Vmin)=Vmin;
        for p=1:k
            if h(j,p)==0
               X(j,p)=X(j,p);
            else
               X(j,p)=res(randperm(numel(res),1));
            end
        end
%         X(j,:)=X(j,:)+V(j,:); 
%         X(j,X(j,:)>popmax)=popmax;
%         X(j,X(j,:)<popmin)=popmin;
%         fitness(j)=fun(X(j,:));  
        fitness(j)=fun1(X(j,:), a, k);
    end 
    for j=1:sizepop        
        if fitness(j)>fitnessPbest(j) 
%         if fitness(j)>fitnessPbest(j)
            Pbest(j,:)=X(j,:);
            fitnessPbest(j)=fitness(j);
%           fitnessPbest(j)=fitnessPbest(j)
        end        
        if fitness(j)>fitnessGbest 
            Gbest=X(j,:);
            fitnessGbest=fitness(j);
%           fitnessGbest=fitnessGbest;
        end
    end
    for j=1:sizepop
      for i1=1:k
         p1=find(a(:,1)==Gbest(i1));
         Neighbors=a2(p1);
         if length(Neighbors)~=0
            X(j,i1)=Neighbors(1);
            fitness1(1)=fun1(X(j,:),a,k);
         end
         if length(p1)>1
           for j1=2:length(p1)
               X(j,i1)=Neighbors(j1);
               fitness1(j1)=fun1(X(j,:),a,k);
               if fitness1(j1)>fitnessGbest&fitness1(j1)>fitness1(j1-1)
                  fitness(j)=fitness1(j1);
               else
                  fitness(j)=fitnessGbest;
               end
           end
         end
      end
     end
    trace(i)=fitnessGbest; 
    disp([Gbest,fitnessGbest]);
end
plot(trace)
title('最优个体适应度');
xlabel('进化代数');
ylabel('适应度');