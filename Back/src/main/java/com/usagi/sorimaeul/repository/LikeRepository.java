package com.usagi.sorimaeul.repository;

import com.usagi.sorimaeul.entity.Like;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface LikeRepository extends JpaRepository<Like, Integer> {

    List<Like> findByUser_userCode(long userCode);

}
