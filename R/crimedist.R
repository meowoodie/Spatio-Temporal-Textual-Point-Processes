# R code for analyzing crime distribution. And plot a pie chart for 
# the analysis result
# 
# By Shixiang Zhu
# Contact: shixiang.zhu@gatech.edu

# basic configurations
root.path           = '/Users/woodie/Desktop/workspace/Event-Series-Detection'
crime.info.path     = paste(root.path, 'data/rawdata/random_cases_info.txt', sep='/')
burglary.set.path   = paste(root.path, 'data/meta/burglary_set.txt', sep='/')
robbery.set.path    = paste(root.path, 'data/meta/robbery_set.txt', sep='/')
# description set for interested crime types
non.crime.set            = c('Miscellaneous Non-Crime', 'School-Miscellaneous Non-Crime', 'NON-CRIMINAL REPORT')
residential.burglary.set = c('Residential Burglary', 'BURG-FORCED ENTRY-RESIDENTIAL')
street.robbery.set       = c('Robbery Street / Other', 'School-ROB-STREET-STRONGARM', 'School-ROB-STREET-OTHER WEAPON',
                             'DOMESTIC VIOLENCE: ROB-STREET-GUN', 'ROB-STREET-KNIFE', 'School-ROB-STREET-GUN', 
                             'DV - ROB-STREET-KNIFE', 'DOMESTIC VIOLENCE: ROB-STREET-STRONGARM', 'DV- ROB-STREET-OTHER WEAPON', 
                             'ROB-STREET-OTHER WEAPON', 'ROB-STREET-GUN', 'ROB-STREET-STRONGARM', 'School-ROB-STREET-KNIFE')
domestic.violent.set     = c('Domestic Violence: DAMAGE TO PROP', 'DOMESTIC VIOLENCE: ROB-MISC-GUN', 'DOMESTIC VIOLENCE: ROB-RESIDENCE-OTH WEAPON',
                             'DOMESTIC VIOLENCE: ROB-STREET-GUN', 'DOMESTIC VIOLENCE: ROB-MISC-STRONGARM', 'DOMESTIC VIOLENCE: ROB-RESIDENCE-GUN',
                             'DOMESTIC VIOLENCE: ROB-RESIDENCE-KNIFE', 'DOMESTIC VIOLENCE: ROB-STREET-STRONGARM', 
                             'DOMESTIC VIOLENCE: ROB-RESIDENCE-STRONGARM', 'DOMESTIC VIOLENCE: ROB-MISC-KNIFE', 
                             'DOMESTIC VIOLENCE: SIMPLE ASLT/BATTERY')

# read raw data into dataframe.
crime.info.df   = read.delim(crime.info.path, header=FALSE, sep='\t')
burglary.set.df = read.delim(burglary.set.path, header=FALSE)
robbery.set.df  = read.delim(robbery.set.path, header=FALSE)
# extract crime description set.
burglary.set    = as.character(burglary.set.df$V1)
robbery.set     = as.character(robbery.set.df$V1)
# extract crime description list.
colnames(crime.info.df) = c('timestamp', 'latitude', 'longitude', 
                            'id', 'categoryCode', 'categoryDesc')   # rename the columns for the dataframe
crime.desc.list = as.character(crime.info.df$categoryDesc)

# calculate 
total.count     = 0
burglary.count  = 0
robbery.count   = 0
residential.burglary.count = 0
street.robbery.count       = 0
for (desc in crime.desc.list) {
  # check burglary and robbery cases
  if (desc %in% burglary.set) {
    burglary.count = burglary.count + 1
  }
  else if (desc %in% robbery.set) {
    robbery.count = robbery.count + 1
  }
  # check subclass
  if (desc %in% residential.burglary.set) {
    residential.burglary.count = residential.burglary.count + 1
  }
  if (desc %in% street.robbery.set) {
    street.robbery.count = street.robbery.count + 1
  }
  # check valid crime cases
  if (!(desc %in% non.crime.set)) {
    total.count = total.count + 1
  }
}
other.count = total.count - burglary.count - robbery.count
other.burglary.count = burglary.count - residential.burglary.count
other.robbery.count  = robbery.count - street.robbery.count
 
slices = c(residential.burglary.count, other.burglary.count, street.robbery.count, other.robbery.count, other.count)
labels = c('Residential Burglary', 'Other Burglary', 'Street Robbery', 'Other Robbery', 'Other crime')
pie(slices, labels=labels, main='Pie Chart of Crime Categories') #, col=colors)

# for burglary and robbery
raw.crime.info.path  = '/Users/woodie/Desktop/workspace/Crime-Pattern-Detection-for-APD/data/records_380k/raw_data.txt'
raw.crime.info.df    = read.delim(raw.crime.info.path, header=FALSE, sep='\t')
notext.crime.info.df = raw.crime.info.df[, c('V1', 'V2', 'V3', 'V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17')]
colnames(notext.crime.info.df) = c(
  'id', 'crime_code', 'crime_desc', 'incident_date', 'ent_datetime', 'upd_datetime',
  'e911_time', 'rec_time', 'disp_time', 'enr_time', 'arv_time', 'transport_time', 
  'booking_time', 'clr_time', 'lat', 'lon', 'text')
notext.crime.info.df = notext.crime.info.df[
  notext.crime.info.df$crime_desc!='',
  # !notext.crime.info.df$crime_desc %in% non.crime.set &
  # !is.na(notext.crime.info.df$rec_time) & 
  # !is.na(notext.crime.info.df$disp_time) &
  # !is.na(notext.crime.info.df$arv_time) &
  # !is.na(notext.crime.info.df$clr_time), 
  c('crime_desc', 'incident_date', 'rec_time', 'disp_time', 'arv_time', 'clr_time',
    'lat', 'lon', 'text')]

convert2time = function (digit.time) {
  secs = digit.time %% 100 
  mins = digit.time %% 10000 %/% 100
  hour = digit.time %/% 10000
  return(secs + 60 * mins + 3600 * hour)
}

check.category = function (descs) {
  categories = c()
  for (desc in descs) {
    if (desc %in% burglary.set) {
      category = 'Burglary'
    }
    else if (desc %in% robbery.set) {
      category = 'Robbery'
    }
    else if (desc %in% domestic.violent.set) {
      category = 'Domestic Violence'
    }
    else {
      category = 'Others'
    }
    categories = c(categories, category)
  }
  return(categories)
}

notext.crime.info.df$rec_time = convert2time(notext.crime.info.df$rec_time)
notext.crime.info.df$disp_time = convert2time(notext.crime.info.df$disp_time)
notext.crime.info.df$arv_time = convert2time(notext.crime.info.df$arv_time)
notext.crime.info.df$clr_time = convert2time(notext.crime.info.df$clr_time)

notext.crime.info.df$t1 = notext.crime.info.df$disp_time - notext.crime.info.df$rec_time
notext.crime.info.df$t2 = notext.crime.info.df$arv_time - notext.crime.info.df$disp_time
notext.crime.info.df$t3 = notext.crime.info.df$clr_time - notext.crime.info.df$arv_time

result.crime.info.df = notext.crime.info.df[
  notext.crime.info.df$t1 != 0 &
  notext.crime.info.df$t2 != 0 &
  notext.crime.info.df$t3 != 0, ]

result.crime.info.df$category = check.category(result.crime.info.df$crime_desc)
library("ggplot2")
ggplot(result.crime.info.df[result.crime.info.df$t1 > 0, ], aes(t1, fill = category)) + 
  geom_density(alpha = .2) +
  xlim(c(0, 10000))

# special version for domestic-violence
notext.crime.info.df = raw.crime.info.df[, c('V1', 'V2', 'V3', 'V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16')]
colnames(notext.crime.info.df) = c(
  'id', 'crime_code', 'crime_desc', 'incident_date', 'ent_datetime', 'upd_datetime',
  'e911_time', 'rec_time', 'disp_time', 'enr_time', 'arv_time', 'transport_time',
  'booking_time', 'clr_time', 'lat', 'lon')
notext.crime.info.df = notext.crime.info.df[
  notext.crime.info.df$crime_desc=='DUI OF ALCOHOL' &
  !notext.crime.info.df$crime_desc %in% non.crime.set &
  !is.na(notext.crime.info.df$arv_time) &
  !is.na(notext.crime.info.df$clr_time),
  c('crime_desc', 'incident_date', 'arv_time', 'clr_time',
    'lat', 'lon')]

# notext.crime.info.df$rec_time = convert2time(notext.crime.info.df$rec_time)
# notext.crime.info.df$disp_time = convert2time(notext.crime.info.df$disp_time)
notext.crime.info.df$arv_time = convert2time(notext.crime.info.df$arv_time)
notext.crime.info.df$clr_time = convert2time(notext.crime.info.df$clr_time)

# notext.crime.info.df$t1 = notext.crime.info.df$disp_time - notext.crime.info.df$rec_time
# notext.crime.info.df$t1 = notext.crime.info.df$arv_time - notext.crime.info.df$disp_time
notext.crime.info.df$t1 = notext.crime.info.df$clr_time - notext.crime.info.df$arv_time

result.crime.info.df = notext.crime.info.df[
  notext.crime.info.df$t1 != 0, ]

ggplot(result.crime.info.df[result.crime.info.df$t1 > 0, ], aes(t1)) +
  geom_density(alpha = 0.2)

