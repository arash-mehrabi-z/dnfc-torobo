function [codestr] = gen_FKcode(M, q, filename)

n = length(q);
if exist('filename','var')        %  filename can be ../xx/yy/zzz/abc.m
    dot = strfind(filename,'.');
    slash = strfind(filename,'/');  % be careful in windows folders
    dot = dot(end);     % last dot   xxxx.m
    if isempty(slash), slash = 1; else, slash = slash(end); end
    functionname = filename(slash+1:dot-1);
else
    functionname = 'computeFK';
end

args='';
for j=1:n
    if (j<n)
        args = sprintf('%s%s, ',args,char(q(j)));
    else
        args = sprintf('%s%s',args,char(q(j)));
    end
end

codestr=sprintf('%% FORWARD KINEMATICS \n');
codestr = sprintf('%s%% Try: T=%s(0,0, 0,0,0,0,0,0,0, 0), rotm2quat(T(1:3,1:3))\n',codestr,functionname);
codestr = sprintf('%sfunction T = %s(%s)\n',codestr,functionname,args);
sprintf('%s    J=zeros(%d,%d);\n', codestr, 3, n);
for i = 1:4
    for j = 1:4
        codestr=sprintf('%s    T(%d,%d) = %s;\n',codestr,i,j,char(vpa(M(i,j))));
    end
end
codestr = sprintf('%s%%end of computeForKin\n', codestr);

if exist('filename','var')
    fid= fopen(filename,'w');
    if (fid>0)
        fprintf(fid, '%s\n',codestr);
        fclose(fid);
        fprintf('Forward kinematic code is written to %s \n', filename);
    else
       fprintf('Forward kinematic file %s cannot be created \n', filename); 
    end
end