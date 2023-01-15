npop=20; 
pc=1;
mu=0.2; 
maxit=100; 
nc=round(pc*npop/2)*2; 
rou=0.2;
usernum=100; 
M0=M(:,1);
M1=unique(M0);
x=M0(:);
x=sort(x);
d=diff([x;max(x)+1]);
count=diff(find([1;d]));
y=[x(find(d)) count]; 
N2=N(:,2);
pg2=pg(:,2); 
empty.position=[]; 
empty.cost=[]; 
empty.rank=[]; 
empty.domination=[]; 
empty.dominated=0; 
empty.crowdingdistance=[];
pop=repmat(empty,npop,1); 
for i=1:npop
  pop(i).position=create_inc(M1, usernum);
  pop(i).cost=costfunction_inc(pop(i).position,y,usernum,M,x,M0,N2,pg2);
end
[pop,F]=nondominatedsort(pop);
pop=calcrowdingdistance(pop,F);
for it=1:maxit
     popc=repmat(empty,nc/2,2);
     pcgen=mu+rou*(it/maxit)^2
     for j=1:nc/2
        p1=tournamentsel(pop);
        p2=tournamentsel(pop);
        [popc(j,1).position,popc(j,2).position]=crossover(p1.position,p2.position);
     end
     popc = popc(:);
     for k = 1 : nc
        popc(k).position=mutate1(popc(k).position,mu,M1,usernum);
        popc(k).cost=costfunction_inc(popc(k).position,y,usernum,M,x,M0,N2,pg2);
     end
     newpop=[pop;popc];
     [pop,F]=nondominatedsort(newpop);
     pop=calcrowdingdistance(pop,F);
     pop=Sortpop(pop);
     pop=pop(1: npop);
     [pop,F]=nondominatedsort(pop);
     pop=calcrowdingdistance(pop,F);
     pop=Sortpop(pop);
     F1=pop(F{1});
     disp(['Iteration' num2str(it) ': Number of F1 Members= ' num2str(numel(F1))])
     figure(1);
     plotcosts(F1);
     pause(0.01);
end