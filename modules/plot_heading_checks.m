function plot_heading_checks(locationame,U,V,allsitebindepths,site_names,site_starttime,site_endtime,allheadings,alldeviceids,numdata4eachmonth)



% fig= figure('units','normalized','position',[0.5,0.5,  .97, .97 ]); %set(fig2, 'Visible', 'off');
%xloc,?,w,bot
% haha=tight_subplot(1,1,[.0530 , 0.052500],[.075075121521250071,.0754812505105],[.41035871052751507571,.410358710527515025]);%set(haha2, 'Visible', 'off');
fig= figure('units','normalized','position',[0,0, 1, 1 ]); %,'visible', 'on');

%xloc,?,w,bot
%                     vert-gap, horz-gap        bot        top
%haha=tight_subplot(2,5,[0.051 , -0.25],[.255121521250071,.074284812505105],[.0007535871052751507571,.0005358710527515025]);%set(haha, 'Visible', 'off');
haha=tight_subplot(2,5,[0.051 , -0.29],[.255121521250071,.074284812505105],[-0.1122962016585058,.1095358710527515025]);%set(haha, 'Visible', 'off');
% set(gcf, 'Visible', 'on');

% -0.111962016585058
% supt=suptitle(locationame);
% supt.Color='g';
% supt. Interpreter='latex';
% supt.FontSize=16;


% expected deployments
expdeps = 1:size(U,3);



% quick check on how many deployments exisit for plotting

actualdeps= expdeps(~ismember(sum(isnan(squeeze(U(1,:,:)))),size(U,2)));

maxXL=nan(10,length(actualdeps));
maxYL=nan(10,length(actualdeps));
minXL=nan(10,length(actualdeps));
minYL=nan(10,length(actualdeps));


% binnumber= 10:-1:1;
binnumber= 1:10;
lwd1=ones(length(actualdeps));
lwd1(end) = 3;

lwd2=.5*ones(length(actualdeps));

clr1 = winter(length(actualdeps));

clr2 = jet(length(actualdeps));

clr2(end,:)=[1.0, 1.0, 1.0];  % current deployment in yellow

axis equal
for dep=1:length(actualdeps)
    
    for jk=1:10
        
        x=squeeze(U(jk,:,dep));y=squeeze(V(jk,:,dep));
        
        % for dynamic axis limits
        
        %         [maxXL(jk,dep), maxYL(:,jk,dep)]= nanmax([x',y']);
        %         [minXL(jk,dep), minYL(:,jk,dep)]= nanmin([x',y']);
        
        [coeff,score,eig_val] = adcp_pca_Mike_to_Ze(x(~isnan(x))',y(~isnan(y))'); %adcp_pca_Mike_to_Ze(u(jk,:)',y');
        eig_val
        OI_princCoord = [0 0; 1 0]*2*sqrt(eig_val(1));
        OJ_princCoord = [0 0; 0 1]*2*sqrt(eig_val(2));
        
        % Generate a current ellipse in principal axes coordinate reference
        % frame scaled to 2_sigma
        rot = -pi:0.01:pi;
        ell = [2*sqrt(eig_val(1))*cos(rot); 2*sqrt(eig_val(2))*sin(rot)]';
        
        % Rotate and translate ellipse and semi-major/minor vectors from
        % principal axis coord ref frame to original coord ref frame of the
        % data
        x_Origin = nanmean(x); y_Origin = nanmean(y);
        oi_trueCoord = OI_princCoord*coeff + repmat([x_Origin y_Origin],2,1);
        oj_trueCoord = OJ_princCoord*coeff + repmat([x_Origin y_Origin],2,1);
        ell_trueCoord = ell*coeff;
        ell_trueCoord(:,1) = ell_trueCoord(:,1) + x_Origin; ell_trueCoord(:,2) = ell_trueCoord(:,2) + y_Origin;
        
        oi_trueCoord_all(:,:,jk) = oi_trueCoord;
        oj_trueCoord_all(:,:,jk) = oj_trueCoord;
        
        % Convert rotation angle of first principal axis to true north (up)
        % because Matlab references it to the x-axis.
        %   coeff consists of the sine and cosine values of the rotation
        %   angle so the sign of these values will determine what compass
        %   quadrant it should be in.
        if (coeff(1,1)>0 && coeff(2,1)>0)   % Quadrant 1
            angle(jk) = round(90 - acosd(coeff(1,1)),1);
        end
        if (coeff(1,1)>0 && coeff(2,1)<0)   % Quadrant 2
            angle(jk) = round(90 + acosd(coeff(1,1)),1);
        end
        if (coeff(1,1)<0 && coeff(2,1)<0)   % Quadrant 3
            angle(jk) = round(90 + acosd(coeff(1,1)),1);
        end
        if (coeff(1,1)<0 && coeff(2,1)>0)   % Quadrant 4
            angle(jk) = round(90 - acosd(coeff(1,1)) + 360,1);
        end
        
        
        axes(haha(jk))
        
        plt(dep)=plot(ell_trueCoord(:,1),ell_trueCoord(:,2),'-','LineWidth',lwd1(dep),'color',clr2(dep,:));
        
        hold all
        plot(oi_trueCoord(:,1),oi_trueCoord(:,2),'--','LineWidth',lwd2(dep),'color',clr2(dep,:)); %text(E_true(2,1),E_true(2,2),num2str(angle),'FontSize',12,'Color','r');
        plot(oj_trueCoord(:,1),oj_trueCoord(:,2),'--','LineWidth',lwd2(dep),'color',clr2(dep,:)); %text(N_true(2,1),N_true(2,2),num2str(angle));
        axis equal
        
        %mytightaxis(findobj('Type','axes'))
        % Label the far left and bottom subplot axes
        % **Note this only works for 5x5 plot panels
        
        if dep ==1
            
            
            
            TL=title(['Bin-',num2str(binnumber(jk))]);
            set(TL,'color','g','interpreter','latex');
            
            ylb=ylabel('v (m s$^{-1}$)');
            set(ylb,'interpreter','latex');
            xlb=xlabel('u (m s$^{-1}$)');
            set(xlb,'interpreter','latex');
        end
        
        %         xlim([-.5 .7]);ylim([-.5 .7]);
%         xlim([-.25 .27]);ylim([-.25 .27]);
        xlim([-.5 .5]);ylim([-.5 .5]);   % FOR bURRARD iNLET
        if ismember(jk, [5,10])==1
            set(gca,'YAxisLocation','right')
        end
        
        if ~ismember(jk, [1,5,6,10])==1
            set(gca,'yticklabel',{})
            ylabel('')
        end
        
        
        if  ismember(jk, [1,2,3,4,5])==1
            set(gca,'xticklabel',{})
            xlabel('')
        end
        
        grid on
        set(gca,'color',.25*[1 1 1],'fontsize',12,'xcolor','g','ycolor','g','zcolor','g','TickLabelInterpreter','latex');
        
        
        
        
        
        
    end
    
    
    
end


set(gcf,'color',.21*[1 1  1]);

% delete(bindepths_info_loc)
bindepths_info_loc=findobj(haha,'Type','axes');

loc=[bindepths_info_loc(6,:).Position(1),  0.25*bindepths_info_loc(6,:).Position(2),  1*bindepths_info_loc(10,:).Position(3),  1*bindepths_info_loc(10,:).Position(4)];

for idx =1:size(allsitebindepths,2)  %1:size(allsitebindepths,2)
    
    bindepths=[allsitebindepths(:,idx)]';
    bindepths=num2str(bindepths,'%10.2f');
    bindepths= [site_names{idx}, ': Bin depths (bin-1, bin-2,...) = ',bindepths];
    
    text_ax=axes('Position',loc);  % log position setting
    %
    sitdepth_loc =.28:-.050:.0051;
    
    if  idx==1
        tx= text(0.35, .35, 'Bin depths (m) for the last five sites:');
        set(tx,'interpreter','latex','color','yellow','fontweight','b','fontsize',14)
        
        tx= text(1.6, .35, 'Heading (deg)');
        set(tx,'interpreter','latex','color','y','fontweight','b','fontsize',14)
        
        tx= text(1.75, .35, 'Device ID:');
        set(tx,'interpreter','latex','color','y','fontweight','b','fontsize',14)
        
        tx= text(1.85, .35, 'Start Date:');
        set(tx,'interpreter','latex','color','y','fontweight','b','fontsize',14)
        
        tx= text(2, .35, 'End Date:');
        set(tx,'interpreter','latex','color','y','fontweight','b','fontsize',14)
        
        
    end
    
    tx= text(0.35, sitdepth_loc(idx), bindepths);
    set(tx,'interpreter','latex','color','w')
    
    tx= text(1.6, sitdepth_loc(idx), num2str(allheadings(idx)));
    set(tx,'interpreter','latex','color','w')
    
    tx= text(1.75, sitdepth_loc(idx), num2str(alldeviceids(idx)));
    set(tx,'interpreter','latex','color','w')
    
    tx= text(1.85, sitdepth_loc(idx), site_starttime{idx});
    set(tx,'interpreter','latex','color','w')
    
    tx= text(2, sitdepth_loc(idx), site_endtime{idx});
    set(tx,'interpreter','latex','color','w')
    
    axis off
    
    
end
box on


lgns = site_names(1:dep);

% figure; plt(1)=plot([0,1],[1,3]);hold all;plt(2)=plot([0,1],[2,4]);lgns={'hi','low'};

gel=legend(plt, lgns);
set(gel,'box','off')
title(gel,'Deployments:','fontsize',26,'interpreter','latex', 'FontWeight','bold')
gel.Title.NodeChildren.Position
gel.TextColor = 'yellow';



set(gel,'location','southeast','box','off','fontsize',16,'color',[0.5 0.5 0.5 ],'interpreter','latex')

%   set(gel,'Position',[gel.Position(1)+.13,gel.Position(2)+.0797911,gel.Position(3), gel.Position(4)],'fontsize',14,'interpreter','latex', 'FontWeight','bold')
%     tx=text(text_ax.Position(1)+.89,text_ax.Position(2)+.5,'Deployments:');
%     set(tx,'color','w','fontsize',16,'interpreter','latex', 'FontWeight','bold')
%x-left                    bot-->up                   width
set(gel,'Position',[text_ax.Position(1)+.89,text_ax.Position(2)+.58,.25*text_ax.Position(3), text_ax.Position(4)],'fontsize',16,'interpreter','latex', 'FontWeight','bold')





%----------------add profile plot--------------

profloc=[0.817087845968711         0.106404958677686         0.148816686722824         0.350206611570249];
prof_ax=axes('Position',profloc);  % log position setting




for dp=1:length(actualdeps)
    xx=squeeze(U(:,:,dp));yy=squeeze(V(:,:,dp));
    
    ang= atan2d(xx,yy);


    
    plot(prof_ax, nanmean(ang,2),allsitebindepths(:,dp),'-','color',clr2(dp,:),'LineWidth',lwd1(dp))
    hold all
    
    if  dp==length(actualdeps)
        
        xlb=xlabel(prof_ax,'current dir. (deg.)');   %'vel. (m s$^{-1}$)');
        set(xlb,'interpreter','latex');
        ylb=ylabel(prof_ax,'depth (m)');
        set(ylb,'interpreter','latex');
        grid on

    end
    
    %----------------------------------------------
end



monthloc=[ 0.817087845968711          0.4750103305785124         0.148816686722824        0.0864325068870524];
monthloc_ax=axes('Position',monthloc);  % log position setting



for dp=1:length(actualdeps)
    IND =numdata4eachmonth(:,dp);
    TOT=nansum(IND);
    
    NN = 100*IND./TOT;
    
    
    plot(monthloc_ax,1:12,NN,'-o','color',clr2(dp,:),'LineWidth',lwd1(dp),'MarkerSize',4,'MarkerFaceColor',clr2(dp,:),'MarkerEdgeColor','none')
    hold all
    if  dp==length(actualdeps)
        
        % plot(monthloc_ax,1:12,100*(N/(sum(N))),'-','color',clr2(dp,:),'LineWidth',lwd1(dep))
        ylb=ylabel(monthloc_ax,{'$\%$ of tot.'; 'data collected'});
        set(ylb,'interpreter','latex');
        xlb=xlabel(monthloc_ax,'Month');
        set(xlb,'interpreter','latex');
        grid on
    end
    
    %----------------------------------------------
end






set(prof_ax,'color',.25*[1 1 1],'fontsize',12,'xcolor','g','ycolor','g','zcolor','g','TickLabelInterpreter','latex','YAxisLoc','right','YDir','rev');

set(monthloc_ax,'color',.25*[1 1 1],'fontsize',10,'xcolor','g','ycolor','g','zcolor','g','TickLabelInterpreter','latex','YAxisLoc','right','XAxisLoc','top','box','on');
set(monthloc_ax,'XLim',[1 12])
%set(monthloc_ax,'YLim',[0 100],'XLim',[1 12])

% gel.Title.NodeChildren.Position = [0.5 1 0];
% gel.Title.NodeChildren.BackgroundColor = 'none';
%gel.TextColor = 'yellow';

% delete(gel)

expression = '(\s+)';
replace = '_';
filename = regexprep(locationame,expression,replace);
outputpath = ['/home/ze/Development_Related/SiteHeadingChecks/',filename,'_Hourly_Mean-Current-Ellipses'];





 export_fig (outputpath,'-r300','-nocrop','-png','-pdf',gcf)
%clf;
%
%
%  % plot(nanmean(v,1)',z,'color','k')   % think about adding this later to the plots
%  delete(monthloc_ax)
% monthloc=[0.769354191736862         0.575413223140498         0.204572803850782         0.128099173553718];
% monthloc_ax=axes('Position',monthloc);  % log position setting
% %
%
% I = checkerboard(10,2,6);
% imshow(I)
% x2 = [2 5; 2 5; 8 8];
% y2 = [4 0; 8 2; 4 0];
% patch(x2,y2,'green')

%
%
% x2 = [0 .1 .1 0 0 0  .1 .1];
% y2 = [0 .1 .1 0 0 0  .2 .2];
% patch(x2,y2,'red')


% for j=1:12
%  plot([j j],[0 j],'k--')
%     hold all
%     axis tight
% end
%
%
% for j=1:5  % dp
%     plot([0,j],[j j],'r--')
%     %axis tight
% end


