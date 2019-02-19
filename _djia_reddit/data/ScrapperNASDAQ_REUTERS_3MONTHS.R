#Input Variables
#Stock_Tickers_NASDAQ<-c("WMT","AAPL","MSFT")
Stock_Tickers_REUTERS<-c("WMT.N","AAPL.OQ","MSFT.OQ","LMT.N","TSCO.L","FDX.N","TSLA.OQ","AMZN.OQ","CVS.N","GM.N","MCK.N","JNJ.N","GOOGL.OQ","FB.OQ")
start_date_ymd_format<-20181115
end_date_ymd_format<-20190215


  
  #Convert the input tickers to be used appropriately and extract data from NASDAQ wesbsite
  Stock_Tickers_NASDAQ<-0
  for(i in 1:length(Stock_Tickers_REUTERS)){
    Stock_Tickers_NASDAQ[i]<-gsub("\\..*","",Stock_Tickers_REUTERS[i])
        print(Stock_Tickers_NASDAQ[i])
  }
  
  
  #Load relevant libraries    
  library(dplyr)
  library(stringi)
  library(xml2)
  library(rvest)
  library(lubridate)
  library(SentimentAnalysis)
  library(ggplot2)
  library(data.table)
  library(gridExtra)
  
  #Download News Articles from Reuters
  
  start_date_ymd_format<-as.Date(as.character(start_date_ymd_format),"%Y%m%d")
  end_date_ymd_format<-as.Date(as.character(end_date_ymd_format),"%Y%m%d")
  days_num<-as.numeric(difftime(end_date_ymd_format, start_date_ymd_format, units = c("auto")))
  date_variable<-seq(start_date_ymd_format, end_date_ymd_format, by="days")
  date_type<-0
  URL_string<-0
  output<-data.frame()
  k<-0
  
  #Build data data frame with variables - Date, Ticker Name and URL; each row represent unique combination Date and Ticker and their corresponding URL; URL is used to extract news
  for(i in 1:length(Stock_Tickers_REUTERS)){
    for(j in 1:length(date_variable)){
      date_type[j]<-paste(
        (if(month(date_variable[j])<10){paste0("0",month(date_variable[j]))}else{month(date_variable[j])}),
        (if(day(date_variable[j])<10){paste0("0",day(date_variable[j]))}else{day(date_variable[j])}),
        year(date_variable[j]),sep = "")
      URL_string[j]<-sprintf("https://www.reuters.com/finance/stocks/company-news/%s?date=%s",Stock_Tickers_REUTERS[i],date_type[j])
      k<-k+1
      output[k,1]<-as.character(date_variable[j])
      output[k,2]<-gsub("\\..*","",Stock_Tickers_REUTERS[i])
      output[k,3]<-as.character(URL_string[j])
    }
  }
  
  #Read News using the URL column of the 'output' dataframe built above
  read_news_headline<-list()
  read_news_abstract<-list()
  for(i in seq_along(output$V3)){
    derby <- read_html(output$V3[i])
    read_news_headline[[i]]<-derby %>%html_nodes(".feature a") %>%html_text()
    read_news_abstract[[i]]<-derby%>%html_nodes("p") %>%html_text()
  }
  
  print("I am here2")
  
  ll<-as.data.frame(stri_list2matrix(read_news_headline, byrow=TRUE))
  ll2<-as.data.frame(stri_list2matrix(read_news_headline, byrow=TRUE))
  
  final_output<-cbind(output,ll2)
  colnames(final_output)[2]<-"Ticker"
  colnames(final_output)[1]<-"Date"
  # final_output<-cbind(output,ll,ll2)
  
  #End Downloading News From Reuters for All Companies
  
  
  #The code below is used to source share price data from NASDAQ
  
  #This section is to create a URL for each company to download stock data from NASDAQ
  URL_string_new<-0
  for(i in 1:length(Stock_Tickers_NASDAQ)){
    URL_string_new[i]<-sprintf("https://www.nasdaq.com/symbol/%s/historical",tolower(Stock_Tickers_NASDAQ[i]))
  }
  
  #This section is to extract information related to share price and store in lists  
  Date_Stock<-list()
  Price_Open<-list()
  Price_High<-list()
  Price_Low<-list()
  Price_Close<-list()
  k<-0
  derby<-list()
  
  for(i in 1:length(Stock_Tickers_NASDAQ)){
    derby <- read_html(URL_string_new[i])
    k=k+1
    Date_Stock[[i]]<-derby %>%html_nodes("td:nth-child(1)") %>%html_text()
    Date_Stock[[i]]<-gsub("[^0-9]", "", Date_Stock[[i]])
    Date_Stock[[i]]<-as.Date(Date_Stock[[i]],"%m%d%Y")
    #    print(Date_Stock[[i]])
    
    Price_Open[[i]]<-derby%>%html_nodes("td:nth-child(2)") %>%html_text()
    Price_Open[[i]]<-gsub("[^[:digit:].]", "", Price_Open[[i]]) 
    
    Price_High[[i]]<-derby%>%html_nodes("td:nth-child(3)") %>%html_text()
    Price_High[[i]]<-gsub("[^[:digit:].]", "", Price_High[[i]]) 
    
    Price_Low[[i]]<-derby%>%html_nodes("td:nth-child(4)") %>%html_text()
    Price_Low[[i]]<-gsub("[^[:digit:].]", "", Price_Low[[i]]) 
    
    Price_Close[[i]]<-derby%>%html_nodes("td:nth-child(5)") %>%html_text()
    Price_Close[[i]]<-gsub("[^[:digit:].]", "", Price_Close[[i]]) 
  }
  print("I am here3")
  
  #This section is to convert data stored in lists into data frames
  date_name<-0
  Date_ll<-as.data.frame(stri_list2matrix(Date_Stock, byrow=FALSE))
  for(i in 1:length(Stock_Tickers_NASDAQ)){
    date_name[i]<-paste("Date_",Stock_Tickers_NASDAQ[i],sep = "")
  }
  colnames(Date_ll)<-date_name
  Date_ll<-na.omit(Date_ll)
  #  print(Date_ll)
  
  Price_Open_ll<-as.data.frame(stri_list2matrix(Price_Open, byrow=FALSE))
  Price_Open_ll$V1<-as.numeric(as.character(Price_Open_ll$V1))
  Price_Open_ll$V2<-as.numeric(as.character(Price_Open_ll$V2))
  Price_Open_ll<-Price_Open_ll[-1,]
  
  Price_Close_ll<-as.data.frame(stri_list2matrix(Price_Close, byrow=FALSE))
  Price_Close_ll$V1<-as.numeric(as.character(Price_Close_ll$V1))
  Price_Close_ll$V2<-as.numeric(as.character(Price_Close_ll$V2))
  Price_Close_ll<-Price_Close_ll[-1,]
  
  Price_High_ll<-as.data.frame(stri_list2matrix(Price_High, byrow=FALSE))
  Price_High_ll$V1<-as.numeric(as.character(Price_High_ll$V1))
  Price_High_ll$V2<-as.numeric(as.character(Price_High_ll$V2))
  Price_High_ll<-Price_High_ll[-1,]
  
  Price_Low_ll<-as.data.frame(stri_list2matrix(Price_Low, byrow=FALSE))
  Price_Low_ll$V1<-as.numeric(as.character(Price_Low_ll$V1))
  Price_Low_ll$V2<-as.numeric(as.character(Price_Low_ll$V2))
  Price_Low_ll<-Price_Low_ll[-1,]
  
  
  #This section is to combine the data stores in data frames into a single data frame with appropriate formatting
  
  ttt<-0
  share_price_outcome<-data.frame()
  
  if(length(Date_ll$Date)!=length(Price_Open_ll$Price_Open)){
    stop("Number of rows collected from the website for date is more than the number of rows of data collected for share price; this may require changes in the code or try wait for an hour and try to see if this is fixed as the code is developed at a particular time in a day")
  } else{
    for(i in 1:length(Stock_Tickers_NASDAQ)){
      for(j in 1:nrow(Date_ll)){
        ttt<-ttt+1
        share_price_outcome[ttt,1]<-as.character(Date_ll[j,i])
        share_price_outcome[ttt,2]<-as.character(Stock_Tickers_NASDAQ[i])
        share_price_outcome[ttt,3]<-as.character(URL_string_new[i])
        share_price_outcome[ttt,4]<-as.numeric(as.character(Price_Open_ll[j,i]))
        share_price_outcome[ttt,5]<-as.numeric(as.character(Price_High_ll[j,i]))
        share_price_outcome[ttt,6]<-as.numeric(as.character(Price_Low_ll[j,i]))
        share_price_outcome[ttt,7]<-as.numeric(as.character(Price_Close_ll[j,i]))
      }
    }
  }
  colnames(share_price_outcome)<-c("Date","Ticker","URL","Price_open","Price_high","Price_low","Price_close")
  share_price_outcome$Ticker<-toupper(share_price_outcome$Ticker)
  share_price_outcome
  
  
  #This section is to combine share price data and news data
  final_data<-merge(share_price_outcome, final_output, by=c("Date","Ticker"))
  
  d<-ncol(final_data)-8
  names_output<-0
  for(i in 1:d)
  {
    names_output[i]<-paste("News",i,sep = "")
  }
  colnames(final_data)<-c("Date","Ticker","URL_Price","Price_open","Price_high","Price_low","Price_close","URL_News",names_output)
  
  
  final_data
  
  ppp<-ncol(final_data)-8
  for (i in 1:ppp){
    i=ppp+8
    final_data[,i] <- gsub('b"',"",final_data[,i])
    final_data[,i] <- gsub("b'","",final_data[,i])
    final_data[,i] <- gsub("[[:punct:]]", "", final_data[,i])
    final_data[,i] <- tolower(final_data[,i])
  }
  
  #Check number of non-NA entries in each column
  #apply(final_data,2,function(x){sum(!is.na(x))})
  print("I am here4")
  #Keep data for companies whoose news in available
  final_output<-final_data[!is.na(final_data$News1),]
  final_output$Date<-as.Date(final_output$Date)
  final_output$Ticker<-factor(final_output$Ticker)
  #Appropriate names of columns and add column on price movement
  print("I am here5")
  
  d<-ncol(final_output)-8
  names_output<-0
  for(i in 1:d)
  {
    names_output[i]<-paste("News",i,sep = "")
  }
  colnames(final_output)<-c("Date","Ticker","URL_Price","Price_open","Price_high","Price_low","Price_close","URL_News",names_output)
  
  print("I am here6")
  
  #Conduct sentiment analysis and store data in data frame named sentiment_u
  k<-0
  c<-list()
  sentiment<-0
  
  for(i in 1:d){
    k=i+8
    c[[i]]<-analyzeSentiment(as.character(final_output[,k]))
  }
  print("I am here7")
  
  sentiment<-rbindlist(c)
  k=nrow(sentiment)/length(c)
  sentiment2<-list()
  
  for(i in 1:length(c)){
    #  print(k*(i-1)+1)
    #  print(k*(i-1)+k)
    sentiment1<-sentiment[((k*(i-1)+1):(k*(i-1)+k)),]
    sentiment2[[i]]<-as.data.frame(convertToDirection(sentiment1$SentimentQDAP))
  }
  
  print("I am here8")
  
  sentiment_u<-bind_cols(sentiment2)
  names_data<-0
  for(i in 1:length(c))
  {
    names_data[i]<-paste("News",i,sep = "")
  }
  colnames(sentiment_u)<-names_data
  
  print("I am here9")
  
  #End of sentiment analysis with a data frame with sentiment score for news articles
  
  
  sentiment_u$positive<-rowSums(sentiment_u=="positive",na.rm = TRUE)
  sentiment_u$negative<-rowSums(sentiment_u=="negative",na.rm = TRUE)
  sentiment_u$neutral<-rowSums(sentiment_u=="neutral",na.rm = TRUE)
  
  
  print("I am here10")
  
  share_price_sentiment<-0
  share_price_sentiment<-as.data.frame(cbind(as.character(final_output$Date),
                                             as.character(final_output$Ticker),
                                             as.character(final_output$URL_Price),
                                             as.character(final_output$URL_News),
                                             as.character(final_output$Price_open),
                                             as.character(final_output$Price_close),
                                             as.numeric(sentiment_u$positive),
                                             as.numeric(sentiment_u$negative),
                                             as.numeric(sentiment_u$neutral)))
  
  print("I am here11")
  
  
  colnames(share_price_sentiment)<-c("Date","Ticker","URL_Price","URL_News",
                                     "Price_open","Price_close","Positive_News",
                                     "Negative_News","Neutral_News")
  
  print("I am here12")
  
  
  share_price_sentiment$Date<-as.Date(as.character(share_price_sentiment$Date))
  share_price_sentiment$Ticker<-as.character(share_price_sentiment$Ticker)
  share_price_sentiment$URL_Price<-as.character(share_price_sentiment$URL_Price)
  share_price_sentiment$URL_News<-as.character(share_price_sentiment$URL_News)
  share_price_sentiment$Price_open<-as.numeric(as.character(share_price_sentiment$Price_open))
  share_price_sentiment$Price_close<-as.numeric(as.character(share_price_sentiment$Price_close))
  share_price_sentiment$Positive_News<-as.numeric(as.character(share_price_sentiment$Positive_News))
  share_price_sentiment$Negative_News<-as.numeric(as.character(share_price_sentiment$Negative_News))
  share_price_sentiment$Neutral_News<-as.numeric(as.character(share_price_sentiment$Neutral_News))
  share_price_sentiment$stock_trend<-as.numeric((share_price_sentiment$Price_open - share_price_sentiment$Price_close)
                                                /share_price_sentiment$Price_open)
  
  print("I am here12")
  
  for(q in 1:length(share_price_sentiment$Ticker)){
    if(as.numeric(share_price_sentiment$stock_trend[q])>=0){
      #      print(q)
      share_price_sentiment$stock_movement[q]="1"
      #      print(share_price_sentiment$stock_movement[q])
    }
    else{
      share_price_sentiment$stock_movement[q]="-1"
    }
  }
  share_price_sentiment$stock_movement<-as.numeric(share_price_sentiment$stock_movement)
  
  print("I am here13")
  
  
  #Plot the share price time series data
  tum<-split(share_price_sentiment,share_price_sentiment$Ticker)
  temp_plot<-list()
  for (i in 1:length(tum)) {
    temp_plot[[i]]<-tum[[i]]%>%ggplot()+
      geom_line(aes(x=Date,y=stock_movement),color="blue")+
      geom_point(aes(x=Date,y=Positive_News),color="red")+
      scale_x_date("%b-%Y")+xlab("Date")+ylab("Stock Price")+
      ggtitle(paste("CompanyName",tum[[i]]$Ticker[1],sep= " "))
    #      scale_shape_discrete(name = "Y Series", labels = c("Stock Movement", "Positive News"))
  }
  print("I am here14")
  
  do.call(grid.arrange, c(temp_plot, ncol = 5))
  share_price_sentiment
  
write.csv(share_price_sentiment, file = "SCRAPTHISSHITFROMNASDAQ201620172018.csv")
  