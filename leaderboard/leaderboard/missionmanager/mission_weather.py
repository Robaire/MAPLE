#!/usr/bin/env python

# Copyright (c) 2023 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Moves the Sun and Earth positions smoothly with time
"""

import datetime
import astropy
import lunarsky

import carla

from leaderboard.utils.timer import GameTime

UPDATE_FREQUENCY = 60            # s

class astrobody():

    def __init__(self, name, datatime, moon_loc):
        
        self.name = name
        t = astropy.time.Time(datatime)
        obj = astropy.coordinates.get_body(name, t)
        obj = obj.transform_to(lunarsky.LunarTopo(location=moon_loc, obstime=datatime))
        self.obj = obj

    def __str__(self):
        return self.name +", "+ str(self.obj)


class Sun(object):

    def __init__(self, world, initial_date, moon_loc):
        self._world = world
        self._weather = world.get_weather()
        self._t = 0.0
        self.initial_date = initial_date
        self.moon_loc = moon_loc

    def update(self, delta_seconds):

        self._t += delta_seconds / 3600
        date = str(self.initial_date + datetime.timedelta(hours=self._t))

        with astropy.coordinates.solar_system_ephemeris.set('builtin'):
            sun = astrobody("sun", date, self.moon_loc)

        self._weather.sun_azimuth_angle = float(sun.obj.az.value)
        self._weather.sun_altitude_angle = float(sun.obj.alt.value)
        self._world.set_weather(self._weather)


class Earth(object):

    UA2KM = 149597871
    SCALE_FACTOR = 0.5  # Scale factor for Earth radius and distance to match the value used in Unreal

    def __init__(self, world, initial_date, moon_loc):

        self.world = world
        self._t = 0.0
        self.initial_date = initial_date
        self.moon_loc = moon_loc
        self.earth_pos_scaled = self.load_and_get_location(initial_date)

        bp = world.get_blueprint_library().filter("*earth*")[0]
        self.earth = world.try_spawn_actor(bp, carla.Transform(carla.Location(x=self.earth_pos_scaled[0],y=self.earth_pos_scaled[1],z=self.earth_pos_scaled[2])))
        newrot = carla.Rotation(roll=180.)
        self.earth.set_transform(carla.Transform(location=self.earth.get_location(),rotation=newrot))

    def load_and_get_location(self, date):

        with astropy.coordinates.solar_system_ephemeris.set('builtin'):
            earth = astrobody("earth", date, self.moon_loc)

        earth.obj = earth.obj.transform_to(lunarsky.MCMF(obstime=date))
        earth_pos = [earth.obj.x.value*self.UA2KM, earth.obj.y.value*self.UA2KM, earth.obj.z.value*self.UA2KM]
        earth_pos_scaled = [coord*self.SCALE_FACTOR for coord in earth_pos]

        return earth_pos_scaled

    def update(self, delta_seconds):

        self._t += delta_seconds / 3600
        date = str(self.initial_date + datetime.timedelta(hours=self._t))

        self.earth_pos_scaled = self.load_and_get_location(date)

        newloc = carla.Location(x=self.earth_pos_scaled[0],y=self.earth_pos_scaled[1],z=self.earth_pos_scaled[2])
        newrot = carla.Rotation(yaw=self._t/24.*360.,roll=180)
        self.earth.set_transform(carla.Transform(newloc,newrot))

    def destroy(self):
        self.earth.destroy()


class MissionWeather(object):

    SPEED = 1
    LATITUDE = -90
    LONGITUDE = 0

    def __init__(self, world):

        self._world = world
        self._delta_seconds = 0

        self._initial_date = datetime.datetime.strptime('2023-01-15 00:00:00', "%Y-%m-%d %H:%M:%S")
        self._moon_loc = lunarsky.MoonLocation(lat=self.LATITUDE,lon=self.LONGITUDE)

        self._sun = Sun(self._world, self._initial_date, self._moon_loc)
        self._earth = Earth(self._world, self._initial_date, self._moon_loc)

        self._sun.update(0)
        self._earth.update(0)

    def tick(self):
        """Update the time and move the sun and earth"""
        timestamp = GameTime.get_time()
        delta_seconds = GameTime.get_delta_time()
        self._delta_seconds += self.SPEED * delta_seconds

        # This tick is very slow, so for performance reasons, make sure this is only
        # executed when there are sensors rendering. TODO: Make this check better
        if not round(timestamp, 2) % UPDATE_FREQUENCY == 0:
            return

        self._sun.update(self._delta_seconds)
        self._earth.update(self._delta_seconds)
        self._delta_seconds = 0

    def cleanup(self):
        """Destroy the earth, literally"""
        self._earth.destroy()
