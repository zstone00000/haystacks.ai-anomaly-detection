import pandas as pd

def import_haystacks_destinations_GA(hstacks_json, list_of_types):
    state = []
    place_id = []
    latitude = []
    longitude = []
    rating = []
    num_ratings = []
    poi_types = []
    name = []

    for j in range(len(hstacks_json)):
        for i in range(len(hstacks_json[j]['responce']['results'])):
            try:
                state_ = hstacks_json[j]['responce']['results'][i]['plus_code']['compound_code'].split(',')[-2].strip()
                poi_types_ = hstacks_json[j]['responce']['results'][i]['types']

                if (state_ == 'GA') and set(poi_types_).intersection(list_of_types):
                    place_id_ = hstacks_json[j]['responce']['results'][i]['place_id']
                    latitude_ = hstacks_json[j]['responce']['results'][i]['geometry']['location']['lat']
                    longitude_ = hstacks_json[j]['responce']['results'][i]['geometry']['location']['lng']
                    rating_ = hstacks_json[j]['responce']['results'][i]['rating']
                    num_ratings_ = hstacks_json[j]['responce']['results'][i]['user_ratings_total']
                    poi_types_ = hstacks_json[j]['responce']['results'][i]['types']
                    name_ = hstacks_json[j]['responce']['results'][i]['name']

                    state.append(state_)
                    place_id.append(place_id_)
                    latitude.append(latitude_)
                    longitude.append(longitude_)
                    rating.append(rating_)
                    num_ratings.append(num_ratings_)
                    poi_types.append(poi_types_)
                    name.append(name_)
            except:
                pass

    return pd.DataFrame({'latitude': latitude, 'longitude': longitude, 'state': state, 'place_id': place_id, 'name': name, 'rating': rating, 'num_ratings': num_ratings, 'poi_types': poi_types})


def import_haystacks_destinations_GA_no_rating(hstacks_json, list_of_types):
    state = []
    place_id = []
    latitude = []
    longitude = []
    poi_types = []
    name = []

    for j in range(len(hstacks_json)):
        for i in range(len(hstacks_json[j]['responce']['results'])):
            try:
                state_ = hstacks_json[j]['responce']['results'][i]['plus_code']['compound_code'].split(',')[-2].strip()
                poi_types_ = hstacks_json[j]['responce']['results'][i]['types']

                if (state_ == 'GA') and set(poi_types_).intersection(list_of_types):
                    place_id_ = hstacks_json[j]['responce']['results'][i]['place_id']
                    latitude_ = hstacks_json[j]['responce']['results'][i]['geometry']['location']['lat']
                    longitude_ = hstacks_json[j]['responce']['results'][i]['geometry']['location']['lng']
                    poi_types_ = hstacks_json[j]['responce']['results'][i]['types']
                    name_ = hstacks_json[j]['responce']['results'][i]['name']

                    state.append(state_)
                    place_id.append(place_id_)
                    latitude.append(latitude_)
                    longitude.append(longitude_)
                    poi_types.append(poi_types_)
                    name.append(name_)
            except:
                pass

    return pd.DataFrame({'latitude': latitude, 'longitude': longitude, 'state': state, 'place_id': place_id, 'name': name, 'poi_types': poi_types})


