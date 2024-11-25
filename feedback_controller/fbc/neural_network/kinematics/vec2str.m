function s = vec2str(q, formatstr)
    if ~exist('formatstr','var')
        formatstr =  '%.4f';
    end
    
    s = '';
    for k=1:length(q)
        elcmd=sprintf('sprintf(''%s'',q(%d))''',formatstr,k);
        elstr= eval(elcmd);
        s= sprintf('%s %s',s, elstr);
    end
    