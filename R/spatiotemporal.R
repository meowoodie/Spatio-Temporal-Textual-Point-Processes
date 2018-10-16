# R code for visualizing spatio-temporal information of crime data
# 
# By Shixiang Zhu
# Contact: shixiang.zhu@gatech.edu

### ---------------------------------------------------------------
### Temporal visualization

library(ggpubr)

# basic configurations
root.path              = '/Users/woodie/Desktop/workspace/Event-Series-Detection'
crime.spatiotempo.path = paste(root.path, 'data/129k.points.txt', sep='/')
crime.label.path       = paste(root.path, 'data/rawdata/labeled_cases_info.txt', sep='/')
crime.random.path      = paste(root.path, 'data/rawdata/random_cases_info.txt', sep='/')
burglary.set.path      = paste(root.path, 'data/meta/burglary_set.txt', sep='/')
robbery.set.path       = paste(root.path, 'data/meta/robbery_set.txt', sep='/')

# read raw data into dataframe.
burglary.set.df = read.delim(burglary.set.path, header=FALSE)
robbery.set.df  = read.delim(robbery.set.path, header=FALSE)
burglary.set    = as.character(burglary.set.df$V1)
robbery.set     = as.character(robbery.set.df$V1)
crime.spatiotempo.df           = read.delim(crime.spatiotempo.path, header=FALSE, sep=',', stringsAsFactors = FALSE)
crime.label.df                 = read.delim(crime.label.path, header=FALSE, sep='\t', stringsAsFactors = FALSE)[,c('V1', 'V2')]
crime.random.df                = read.delim(crime.random.path, header=FALSE, sep='\t', stringsAsFactors = FALSE)[,c('V4', 'V6')]
colnames(crime.label.df)       = c('id', 'Labels')
colnames(crime.random.df)      = c('id', 'Labels')
crime.spatiotempo.info.df      = rbind(crime.label.df, crime.random.df)
colnames(crime.spatiotempo.df) = c('timestamp', 'lat', 'lon')   # rename the columns for the dataframe
labeld.crime.spatiotempo.df    = crime.spatiotempo.df[1:56,]
crime.spatiotempo.complete.df  = cbind(crime.spatiotempo.info.df, crime.spatiotempo.df)
residential.burglary.set = c('Residential Burglary', 'BURG-FORCED ENTRY-RESIDENTIAL')
street.robbery.set       = c('Robbery Street / Other', 'School-ROB-STREET-STRONGARM', 'School-ROB-STREET-OTHER WEAPON',
                             'DOMESTIC VIOLENCE: ROB-STREET-GUN', 'ROB-STREET-KNIFE', 'School-ROB-STREET-GUN', 
                             'DV - ROB-STREET-KNIFE', 'DOMESTIC VIOLENCE: ROB-STREET-STRONGARM', 'DV- ROB-STREET-OTHER WEAPON', 
                             'ROB-STREET-OTHER WEAPON', 'ROB-STREET-GUN', 'ROB-STREET-STRONGARM', 'School-ROB-STREET-KNIFE')

# crime.spatiotempo.partial.df     = crime.spatiotempo.complete.df[crime.spatiotempo.complete.df$Labels %in% residential.burglary.set | crime.spatiotempo.complete.df$Labels %in% street.robbery.set, ]
# crime.spatiotempo.partial.df$tag = crime.spatiotempo.partial.df$Labels %in% residential.burglary.set
# 
# ggdensity(crime.spatiotempo.partial.df, x = 'timestamp', rug = TRUE,
#           color = 'tag', fill = 'tag', ylab = 'Intensity', xlab = 'Unix timestamp',
#           palette = c("#4682b4", "#aaaebc"))

# customize dataset for better visualization
labeld.crime.df = cbind(crime.label.df, labeld.crime.spatiotempo.df)
labeld.crime.df[23, 'timestamp'] = 1484123800 # 1484523800
labeld.crime.df[26, 'timestamp'] = 1484939400 # 1484439400
# refine the labels
for (i in 1:56) {
  if (labeld.crime.df[i, 'Labels'] == 'burglary') {
    labeld.crime.df[i, 'Labels'] = 'Burglary in Buckhead'
  }
  else if (labeld.crime.df[i, 'Labels'] == 'pedrobbery') {
    labeld.crime.df[i, 'Labels'] = 'Ped Robbery in Buckhead'
  }
  else if (labeld.crime.df[i, 'Labels'] == 'DIJAWAN_ADAMS') {
    labeld.crime.df[i, 'Labels'] = 'Ped Robbery by A'
  }
  else if (labeld.crime.df[i, 'Labels'] == 'JAYDARIOUS_MORRISON') {
    labeld.crime.df[i, 'Labels'] = 'Ped Robbery by M'
  }
  else if (labeld.crime.df[i, 'Labels'] == 'JULIAN_TUCKER') {
    labeld.crime.df[i, 'Labels'] = 'Ped Robbery by J'
  }
  else if (labeld.crime.df[i, 'Labels'] == 'THADDEUS_TODD') {
    labeld.crime.df[i, 'Labels'] = 'Ped Robbery by T'
  }
}

# Change outline and fill colors by groups ("sex")
# Use a custom palette
ggdensity(labeld.crime.df, x = 'timestamp', rug = TRUE,
          color = 'Labels', fill = 'Labels', ylab = 'Intensity', xlab = 'Unix timestamp',
          palette = c("#4682b4", "#aaaebc", "#cc98e5", "#ed7f00", "#518251", "#be4a47"))




### ---------------------------------------------------------------
### Spatial visualization

library(viridis)

# if(!requireNamespace("devtools")) install.packages("devtools")
# devtools::install_github("dkahle/ggmap", ref = "tidyup")
# ggmap(get_googlemap())

if(!requireNamespace("devtools")) install.packages("devtools")
devtools::install_github("dkahle/ggmap", ref = "tidyup")
register_google(key="AIzaSyBhqJ7iF3b_cYbt8Q4-XYE7ZnDVUhGJax0")

crime.spatiotempo.burglary.df = crime.spatiotempo.complete.df[crime.spatiotempo.complete.df$Labels %in% residential.burglary.set, ]
crime.spatiotempo.robbery.df = crime.spatiotempo.complete.df[crime.spatiotempo.complete.df$Labels %in% street.robbery.set, ]

basemap1 = get_map(location = c(lon = -84.3880, lat = 33.7530), zoom = 12, maptype = "toner-lite")
ggmap(basemap1) +
  stat_density2d(aes(fill = ..level..), alpha = .5, 
                 h = .01,
                 geom = "polygon", data = crime.spatiotempo.burglary.df) + 
  scale_fill_viridis() + 
  theme(legend.position = 'none')

basemap2 = get_map(location = c(lon = -84.4000, lat = 33.8500), zoom = 12, maptype = "toner-lite")
ggmap(basemap2) +
  # geom_point(data=labeld.crime.df[labeld.crime.df$Labels=='Burglary in Buckhead', ], col="red", size=.5) + 
  stat_density2d(aes(fill = ..level..), alpha = .5, h = .04,
                 geom = "polygon", data = labeld.crime.df[labeld.crime.df$Labels=='Burglary in Buckhead', ]) + 
  scale_fill_viridis() + 
  theme(legend.position = 'none')

basemap2 = get_map(location = c(lon = -84.3880, lat = 33.7530), zoom = 11, maptype = "toner-lite")
ggmap(basemap2) +
  # geom_point(data=labeld.crime.df[labeld.crime.df$Labels=='Ped Robbery by M', ], col="red", size=.5)
  stat_density2d(aes(fill = ..level..), alpha = .5, h = .05,
                 geom = "polygon", data = labeld.crime.df[labeld.crime.df$Labels=='Ped Robbery in Buckhead', ]) +
  scale_fill_viridis() +
  theme(legend.position = 'none')

