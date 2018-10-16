# R code for analyzing crime distribution. And plot a pie chart for 
# the analysis result
# 
# By Shixiang Zhu
# Contact: shixiang.zhu@gatech.edu

# basic configurations
root.path         = '/Users/woodie/Desktop/workspace/Event-Series-Detection'
crime.info.path   = paste(root.path, 'data/rawdata/random_cases_info.txt', sep='/')
burglary.set.path = paste(root.path, 'data/meta/burglary_set.txt', sep='/')
robbery.set.path  = paste(root.path, 'data/meta/robbery_set.txt', sep='/')
# description set for interested crime types
non.crime.set            = c('Miscellaneous Non-Crime', 'School-Miscellaneous Non-Crime', 'NON-CRIMINAL REPORT')
residential.burglary.set = c('Residential Burglary', 'BURG-FORCED ENTRY-RESIDENTIAL')
street.robbery.set       = c('Robbery Street / Other', 'School-ROB-STREET-STRONGARM', 'School-ROB-STREET-OTHER WEAPON',
                             'DOMESTIC VIOLENCE: ROB-STREET-GUN', 'ROB-STREET-KNIFE', 'School-ROB-STREET-GUN', 
                             'DV - ROB-STREET-KNIFE', 'DOMESTIC VIOLENCE: ROB-STREET-STRONGARM', 'DV- ROB-STREET-OTHER WEAPON', 
                             'ROB-STREET-OTHER WEAPON', 'ROB-STREET-GUN', 'ROB-STREET-STRONGARM', 'School-ROB-STREET-KNIFE')


# read raw data into dataframe.
crime.info.df   = read.delim(crime.info.path, header=FALSE, sep='\t')
burglary.set.df = read.delim(burglary.set.path, header=FALSE)
robbery.set.df  = read.delim(robbery.set.path, header=FALSE)
# extract crime description list.
colnames(crime.info.df) = c('timestamp', 'latitude', 'longitude', 
                            'id', 'categoryCode', 'categoryDesc')   # rename the columns for the dataframe
crime.desc.list = as.character(crime.info.df$categoryDesc)
# extract crime description set.
burglary.set    = as.character(burglary.set.df$V1)
robbery.set     = as.character(robbery.set.df$V1)

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


